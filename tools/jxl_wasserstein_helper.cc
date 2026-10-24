// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/cms.h>
#include <jxl/memory_manager.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "lib/extras/codec.h"
#include "lib/extras/dec/color_hints.h"
#include "lib/extras/dec/decode.h"
#include "lib/extras/packed_image.h"
#include "lib/extras/packed_image_convert.h"
#include "lib/jxl/ac_context.h"
#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/override.h"
#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/chroma_from_luma.h"
#include "lib/extras/codec_in_out.h"
#include "lib/jxl/coeff_order.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/common.h"
#include "lib/jxl/compressed_dc.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/dec_cache.h"
#include "lib/jxl/dec_frame.h"
#include "lib/jxl/dec_modular.h"
#include "lib/jxl/dec_patch_dictionary.h"
#include "lib/jxl/enc_adaptive_quantization.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_chroma_from_luma.h"
#include "lib/jxl/enc_coeff_order.h"
#include "lib/jxl/enc_context_map.h"
#include "lib/jxl/enc_entropy_coder.h"
#include "lib/jxl/enc_fields.h"
#include "lib/jxl/enc_frame.h"
#include "lib/jxl/enc_group.h"
#include "lib/jxl/enc_heuristics.h"
#include "lib/jxl/enc_modular.h"
#include "lib/jxl/enc_noise.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/enc_patch_dictionary.h"
#include "lib/jxl/enc_quant_weights.h"
#include "lib/jxl/enc_splines.h"
#include "lib/jxl/enc_toc.h"
#include "lib/jxl/enc_xyb.h"
#include "lib/jxl/epf.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/headers.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/loop_filter.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/noise.h"
#include "lib/jxl/passes_state.h"
#include "lib/jxl/quant_weights.h"
#include "lib/jxl/quantizer.h"
#include "lib/jxl/toc.h"
#include "tools/file_io.h"
#include "tools/no_memory_manager.h"
#include "tools/thread_pool_internal.h"

namespace jxl {

void FindBestBlockEntropyModel(const CompressParams& cparams, const ImageI& rqf,
                               const AcStrategyImage& ac_strategy,
                               BlockCtxMap* block_ctx_map);

namespace {

#pragma pack(push, 1)
struct JxlStateHeader {
  char magic[8];                // "JXLWST01"
  uint32_t version;             // 1
  uint32_t height;              // H
  uint32_t width;               // W
  uint32_t h8;                  // H8 = DivCeil(H, 8)
  uint32_t w8;                  // W8 = DivCeil(W, 8)
  uint32_t ht;                  // HT = DivCeil(H8, 8)
  uint32_t wt;                  // WT = DivCeil(W8, 8)
  int32_t global_scale;
  int32_t quant_dc;
  int32_t x_qm_scale;
  int32_t b_qm_scale;
  int32_t color_factor;
  float base_correlation_x;
  float base_correlation_b;
  int32_t ytox_dc;
  int32_t ytob_dc;
  int32_t epf_iters;
  float epf_quant_mul;
  float epf_pass0_sigma_scale;
  float epf_pass2_sigma_scale;
  float epf_border_sad_mul;
  float epf_channel_scale[3];
  float epf_sharp_lut[8];
  uint32_t gab;
  float gab_weights[6];
  uint32_t used_acs;
  uint32_t has_dequant_tables;
  uint64_t total_bits;
  uint64_t header_bits;
  uint64_t dc_bits;
  uint64_t ac_meta_bits;
  uint64_t ac_token_bits;
  uint64_t file_bytes;
  uint64_t layer_bits[15];
};
static_assert(sizeof(JxlStateHeader) == 340, "JxlStateHeader must be 340 bytes");

struct JxlPatchesHeader {
  char magic[8];        // "JXLWPT01"
  uint32_t version;     // 1
  uint32_t num_patches; // N
};
static_assert(sizeof(JxlPatchesHeader) == 16,
              "JxlPatchesHeader must be 16 bytes");

struct JxlPatchEntry {
  uint32_t ref;
  uint32_t x0;
  uint32_t y0;
  uint32_t xsize;
  uint32_t ysize;
  uint32_t x;
  uint32_t y;
  uint32_t blend_mode;  // 2 = kAdd (or 4..7 for alpha / alphamuladd)
};
static_assert(sizeof(JxlPatchEntry) == 32, "JxlPatchEntry must be 32 bytes");

struct JxlAlphaHeader {
  char magic[8];      // "JXLWAL01"
  uint32_t version;   // 1
  uint32_t height;    // H_alpha
  uint32_t width;     // W_alpha
  uint32_t dim_shift; // D in {0, 1, 2, 3}
};
static_assert(sizeof(JxlAlphaHeader) == 24, "JxlAlphaHeader must be 24 bytes");
#pragma pack(pop)

// Standard C++17 explicit instantiation idiom to read PatchDictionary private
// vectors in --inspect without modifying lib/jxl/dec_patch_dictionary.h.
template <typename Tag, typename Tag::type M>
struct AccessPrivate {
  friend typename Tag::type GetPrivate(Tag) { return M; }
};
struct PatchPositionsTag {
  using type = std::vector<PatchPosition> PatchDictionary::*;
  friend type GetPrivate(PatchPositionsTag);
};
struct PatchRefPositionsTag {
  using type = std::vector<PatchReferencePosition> PatchDictionary::*;
  friend type GetPrivate(PatchRefPositionsTag);
};
struct PatchBlendingsTag {
  using type = std::vector<PatchBlending> PatchDictionary::*;
  friend type GetPrivate(PatchBlendingsTag);
};
struct PatchBlendingsStrideTag {
  using type = size_t PatchDictionary::*;
  friend type GetPrivate(PatchBlendingsStrideTag);
};
template struct AccessPrivate<PatchPositionsTag, &PatchDictionary::positions_>;
template struct AccessPrivate<PatchRefPositionsTag,
                              &PatchDictionary::ref_positions_>;
template struct AccessPrivate<PatchBlendingsTag, &PatchDictionary::blendings_>;
template struct AccessPrivate<PatchBlendingsStrideTag,
                              &PatchDictionary::blendings_stride_>;

const std::vector<PatchPosition>& GetPatchPositions(
    const PatchDictionary& pdic) {
  return pdic.*GetPrivate(PatchPositionsTag{});
}
const std::vector<PatchReferencePosition>& GetPatchRefPositions(
    const PatchDictionary& pdic) {
  return pdic.*GetPrivate(PatchRefPositionsTag{});
}
const std::vector<PatchBlending>& GetPatchBlendings(
    const PatchDictionary& pdic) {
  return pdic.*GetPrivate(PatchBlendingsTag{});
}
size_t GetPatchBlendingsStride(const PatchDictionary& pdic) {
  return pdic.*GetPrivate(PatchBlendingsStrideTag{});
}

static constexpr size_t kAcsCoveredBlocksX[27] = {
    1, 1, 1, 1, 2, 4, 1, 2, 1, 4, 2, 4, 1, 1,
    1, 1, 1, 1, 8, 4, 8, 16, 8, 16, 32, 16, 32,
};
static constexpr size_t kAcsCoveredBlocksY[27] = {
    1, 1, 1, 1, 2, 4, 2, 1, 4, 1, 4, 2, 1, 1,
    1, 1, 1, 1, 8, 8, 4, 16, 16, 8, 32, 32, 16,
};

static const char* const kLayerNames[15] = {
    "Header",         "Toc",           "Dictionary",   "Splines",
    "Noise",          "Quant",         "ModularTree",  "ModularGlobal",
    "Dc",             "ModularDcGroup","ControlFields","Order",
    "Ac",             "AcTokens",      "ModularAcGroup",
};

struct LoadedState {
  JxlStateHeader hdr{};
  std::vector<uint8_t> ac_strategy;
  std::vector<int32_t> raw_quant_field;
  std::vector<uint8_t> epf_sharpness;
  std::vector<int8_t> ytox_map;
  std::vector<int8_t> ytob_map;
  std::vector<int32_t> dc_quant;
  std::vector<int32_t> quant_ac;
  bool has_custom_dequant = false;
  bool force_custom_dequant = false;
  std::vector<float> custom_dequant_matrices[AcStrategy::kNumValidStrategies];
  bool has_custom_dc_quant = false;
  float custom_dc_quant[3] = {kDCQuant[0], kDCQuant[1], kDCQuant[2]};
};

struct PixelStatsForChromacityAdjustment {
  float dx = 0;
  float db = 0;
  float exposed_blue = 0;
  static float CalcPlane(const ImageF* JXL_RESTRICT plane, const Rect& rect) {
    float xmax = 0;
    float ymax = 0;
    for (size_t ty = 1; ty < rect.ysize(); ++ty) {
      for (size_t tx = 1; tx < rect.xsize(); ++tx) {
        float cur = rect.Row(plane, ty)[tx];
        float prev_row = rect.Row(plane, ty - 1)[tx];
        float prev = rect.Row(plane, ty)[tx - 1];
        xmax = std::max(xmax, std::abs(cur - prev));
        ymax = std::max(ymax, std::abs(cur - prev_row));
      }
    }
    return std::max(xmax, ymax);
  }
  void CalcExposedBlue(const ImageF* JXL_RESTRICT plane_y,
                       const ImageF* JXL_RESTRICT plane_b, const Rect& rect) {
    float eb = 0;
    float xmax = 0;
    float ymax = 0;
    for (size_t ty = 1; ty < rect.ysize(); ++ty) {
      for (size_t tx = 1; tx < rect.xsize(); ++tx) {
        float cur_y = rect.Row(plane_y, ty)[tx];
        float cur_b = rect.Row(plane_b, ty)[tx];
        float exposed_b = cur_b - cur_y * 1.2f;
        float diff_b = cur_b - cur_y;
        float prev_row = rect.Row(plane_b, ty - 1)[tx];
        float prev = rect.Row(plane_b, ty)[tx - 1];
        float diff_prev_row = prev_row - rect.Row(plane_y, ty - 1)[tx];
        float diff_prev = prev - rect.Row(plane_y, ty)[tx - 1];
        xmax = std::max(xmax, std::abs(diff_b - diff_prev));
        ymax = std::max(ymax, std::abs(diff_b - diff_prev_row));
        if (exposed_b >= 0) {
          exposed_b *= std::abs(cur_b - prev) + std::abs(cur_b - prev_row);
          eb = std::max(eb, exposed_b);
        }
      }
    }
    exposed_blue = eb;
    db = std::max(xmax, ymax);
  }
  void Calc(const Image3F* JXL_RESTRICT opsin, const Rect& rect) {
    dx = CalcPlane(&opsin->Plane(0), rect);
    CalcExposedBlue(&opsin->Plane(1), &opsin->Plane(2), rect);
  }
  int HowMuchIsXChannelPixelized() const {
    if (dx >= 0.026f) return 3;
    if (dx >= 0.022f) return 2;
    if (dx >= 0.015f) return 1;
    return 0;
  }
  int HowMuchIsBChannelPixelized() const {
    int add = exposed_blue >= 0.13f ? 1 : 0;
    if (db > 0.38f) return 2 + add;
    if (db > 0.33f) return 1 + add;
    if (db > 0.28f) return add;
    return 0;
  }
};

void ComputeChromacityAdjustments(const CompressParams& cparams,
                                  const Image3F& opsin, const Rect& rect,
                                  FrameHeader* frame_header) {
  if (frame_header->encoding != FrameEncoding::kVarDCT ||
      cparams.max_error_mode) {
    return;
  }
  float x_qm_scale_steps[3] = {2.5f, 5.5f, 9.5f};
  frame_header->x_qm_scale = 3;
  for (float x_qm_scale_step : x_qm_scale_steps) {
    if (cparams.original_butteraugli_distance > x_qm_scale_step) {
      frame_header->x_qm_scale++;
    }
  }
  PixelStatsForChromacityAdjustment pixel_stats;
  if (cparams.speed_tier <= SpeedTier::kSquirrel) {
    pixel_stats.Calc(&opsin, rect);
  }
  frame_header->x_qm_scale = std::max<int>(
      frame_header->x_qm_scale, 2 + pixel_stats.HowMuchIsXChannelPixelized());
  frame_header->b_qm_scale = 2 + pixel_stats.HowMuchIsBChannelPixelized();
}

Status LoopFilterFromParams(const CompressParams& cparams,
                            FrameHeader* JXL_RESTRICT frame_header) {
  LoopFilter* loop_filter = &frame_header->loop_filter;
  loop_filter->gab = ApplyOverride(
      cparams.gaborish, cparams.speed_tier <= SpeedTier::kHare &&
                            frame_header->encoding == FrameEncoding::kVarDCT &&
                            cparams.decoding_speed_tier < 4 &&
                            cparams.butteraugli_distance > 0.5f &&
                            !cparams.disable_perceptual_optimizations);
  if (cparams.epf != -1) {
    loop_filter->epf_iters = cparams.epf;
  } else if (cparams.disable_perceptual_optimizations ||
             frame_header->encoding == FrameEncoding::kModular) {
    loop_filter->epf_iters = 0;
  } else {
    constexpr float kThresholds[3] = {0.7f, 1.5f, 4.0f};
    loop_filter->epf_iters = 0;
    if (cparams.decoding_speed_tier < 3) {
      for (size_t i = cparams.decoding_speed_tier == 2 ? 1 : 0; i < 3; i++) {
        if (cparams.butteraugli_distance >= kThresholds[i]) {
          loop_filter->epf_iters++;
        }
      }
    }
  }
  return true;
}

Status ComputeAllCoeffOrders(PassesEncoderState& enc_state,
                             const FrameDimensions& frame_dim) {
  auto used_orders_info = ComputeUsedOrders(
      enc_state.cparams.speed_tier, enc_state.shared.ac_strategy,
      Rect(enc_state.shared.raw_quant_field));
  enc_state.used_orders.resize(enc_state.progressive_splitter.GetNumPasses());
  for (size_t i = 0; i < enc_state.progressive_splitter.GetNumPasses(); i++) {
    JXL_RETURN_IF_ERROR(ComputeCoeffOrder(
        enc_state.cparams.speed_tier, *enc_state.coeffs[i],
        enc_state.shared.ac_strategy, frame_dim, enc_state.used_orders[i],
        enc_state.used_acs, used_orders_info.first, used_orders_info.second,
        &enc_state.shared.coeff_orders[i * enc_state.shared.coeff_order_size]));
  }
  enc_state.used_acs |= used_orders_info.first;
  return true;
}

struct EncCache {
  Status InitOnce(JxlMemoryManager* memory_manager) {
    if (num_nzeroes.xsize() == 0) {
      JXL_ASSIGN_OR_RETURN(num_nzeroes,
                           Image3I::Create(memory_manager, kGroupDimInBlocks,
                                           kGroupDimInBlocks));
    }
    return true;
  }
  Image3I num_nzeroes;
};

Status TokenizeAllCoefficients(const FrameHeader& frame_header,
                               ThreadPool* pool,
                               PassesEncoderState* enc_state) {
  PassesSharedState& shared = enc_state->shared;
  std::vector<EncCache> group_caches;
  JxlMemoryManager* memory_manager = enc_state->memory_manager();
  const auto tokenize_group_init = [&](const size_t num_threads) -> Status {
    group_caches.resize(num_threads);
    return true;
  };
  const auto tokenize_group = [&](const uint32_t group_index,
                                  const size_t thread) -> Status {
    const Rect rect = shared.frame_dim.BlockGroupRect(group_index);
    for (size_t idx_pass = 0; idx_pass < enc_state->passes.size(); idx_pass++) {
      JXL_ENSURE(enc_state->coeffs[idx_pass]->Type() == ACType::k32);
      const int32_t* JXL_RESTRICT ac_rows[3] = {
          enc_state->coeffs[idx_pass]->PlaneRow(0, group_index, 0).ptr32,
          enc_state->coeffs[idx_pass]->PlaneRow(1, group_index, 0).ptr32,
          enc_state->coeffs[idx_pass]->PlaneRow(2, group_index, 0).ptr32,
      };
      JXL_RETURN_IF_ERROR(group_caches[thread].InitOnce(memory_manager));
      JXL_RETURN_IF_ERROR(TokenizeCoefficients(
          &shared.coeff_orders[idx_pass * shared.coeff_order_size], rect,
          ac_rows, shared.ac_strategy, frame_header.chroma_subsampling,
          &group_caches[thread].num_nzeroes,
          &enc_state->passes[idx_pass].ac_tokens[group_index], shared.quant_dc,
          shared.raw_quant_field, shared.block_ctx_map));
    }
    return true;
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, shared.frame_dim.num_groups,
                                tokenize_group_init, tokenize_group,
                                "TokenizeGroup"));
  return true;
}

Status EncodeGlobalDCInfo(const PassesSharedState& shared, BitWriter* writer,
                          AuxOut* aux_out) {
  QuantizerParams params = shared.quantizer.GetParams();
  JXL_RETURN_IF_ERROR(
      WriteQuantizerParams(params, writer, LayerType::Quant, aux_out));
  JXL_RETURN_IF_ERROR(EncodeBlockCtxMap(shared.block_ctx_map, writer, aux_out));
  JXL_RETURN_IF_ERROR(ColorCorrelationEncodeDC(shared.cmap.base(), writer,
                                               LayerType::Dc, aux_out));
  return true;
}

Status EncodeGlobalACInfo(PassesEncoderState* enc_state, BitWriter* writer,
                          ModularFrameEncoder* enc_modular, AuxOut* aux_out) {
  PassesSharedState& shared = enc_state->shared;
  JxlMemoryManager* memory_manager = enc_state->memory_manager();
  JXL_RETURN_IF_ERROR(DequantMatricesEncode(memory_manager, shared.matrices,
                                            writer, LayerType::Quant, aux_out,
                                            enc_modular));
  size_t num_histo_bits = CeilLog2Nonzero(shared.frame_dim.num_groups);
  if (!enc_state->streaming_mode && num_histo_bits != 0) {
    JXL_RETURN_IF_ERROR(
        writer->WithMaxBits(num_histo_bits, LayerType::Ac, aux_out, [&] {
          writer->Write(num_histo_bits, shared.num_histograms - 1);
          return true;
        }));
  }

  for (size_t i = 0; i < enc_state->progressive_splitter.GetNumPasses(); i++) {
    if (!enc_state->streaming_mode) {
      size_t order_bits = 0;
      JXL_RETURN_IF_ERROR(U32Coder::CanEncode(
          kOrderEnc, enc_state->used_orders[i], &order_bits));
      JXL_RETURN_IF_ERROR(
          writer->WithMaxBits(order_bits, LayerType::Order, aux_out, [&] {
            return U32Coder::Write(kOrderEnc, enc_state->used_orders[i],
                                   writer);
          }));
      JXL_RETURN_IF_ERROR(
          EncodeCoeffOrders(enc_state->used_orders[i],
                            &shared.coeff_orders[i * shared.coeff_order_size],
                            writer, LayerType::Order, aux_out));
    }

    HistogramParams hist_params(enc_state->cparams.speed_tier,
                                shared.block_ctx_map.NumACContexts());
    if (enc_state->cparams.speed_tier > SpeedTier::kTortoise) {
      hist_params.lz77_method = HistogramParams::LZ77Method::kNone;
    }
    if (enc_state->cparams.decoding_speed_tier >= 1) {
      hist_params.max_histograms = 6;
    }
    size_t num_histogram_groups = shared.num_histograms;
    hist_params.streaming_mode = enc_state->streaming_mode;
    hist_params.initialize_global_state = enc_state->initialize_global_state;
    JXL_ASSIGN_OR_RETURN(
        size_t cost,
        BuildAndEncodeHistograms(
            memory_manager, hist_params,
            num_histogram_groups * shared.block_ctx_map.NumACContexts(),
            enc_state->passes[i].ac_tokens, &enc_state->passes[i].codes, writer,
            LayerType::Ac, aux_out));
    (void)cost;
  }

  return true;
}

Status EncodeGroups(const FrameHeader& frame_header,
                    PassesEncoderState* enc_state,
                    ModularFrameEncoder* enc_modular, ThreadPool* pool,
                    std::vector<std::unique_ptr<BitWriter>>* group_codes,
                    AuxOut* aux_out) {
  const PassesSharedState& shared = enc_state->shared;
  JxlMemoryManager* memory_manager = shared.memory_manager;
  const FrameDimensions& frame_dim = shared.frame_dim;
  const size_t num_groups = frame_dim.num_groups;
  const size_t num_passes = enc_state->progressive_splitter.GetNumPasses();
  const size_t global_ac_index = frame_dim.num_dc_groups + 1;
  const bool is_small_image =
      !enc_state->streaming_mode && num_groups == 1 && num_passes == 1;
  const size_t num_toc_entries =
      is_small_image ? 1
                     : AcGroupIndex(0, 0, num_groups, frame_dim.num_dc_groups) +
                           num_groups * num_passes;
  JXL_ENSURE(group_codes->empty());
  group_codes->reserve(num_toc_entries);
  for (size_t i = 0; i < num_toc_entries; ++i) {
    group_codes->emplace_back(jxl::make_unique<BitWriter>(memory_manager));
  }

  const auto get_output = [&](const size_t index) -> BitWriter* {
    return (*group_codes)[is_small_image ? 0 : index].get();
  };
  auto ac_group_code = [&](size_t pass, size_t group) {
    return get_output(AcGroupIndex(pass, group, frame_dim.num_groups,
                                   frame_dim.num_dc_groups));
  };

  if (enc_state->initialize_global_state) {
    if (frame_header.flags & FrameHeader::kPatches) {
      JXL_RETURN_IF_ERROR(PatchDictionaryEncoder::Encode(
          shared.image_features.patches, get_output(0), LayerType::Dictionary,
          aux_out));
    }
    if (frame_header.flags & FrameHeader::kSplines) {
      JXL_RETURN_IF_ERROR(EncodeSplines(shared.image_features.splines,
                                        get_output(0), LayerType::Splines,
                                        HistogramParams(), aux_out));
    }
    if (frame_header.flags & FrameHeader::kNoise) {
      JXL_RETURN_IF_ERROR(EncodeNoise(shared.image_features.noise_params,
                                      get_output(0), LayerType::Noise,
                                      aux_out));
    }

    JXL_RETURN_IF_ERROR(DequantMatricesEncodeDC(shared.matrices, get_output(0),
                                                LayerType::Quant, aux_out));
    if (frame_header.encoding == FrameEncoding::kVarDCT) {
      JXL_RETURN_IF_ERROR(EncodeGlobalDCInfo(shared, get_output(0), aux_out));
    }
    JXL_RETURN_IF_ERROR(enc_modular->EncodeGlobalInfo(enc_state->streaming_mode,
                                                      get_output(0), aux_out));
    JXL_RETURN_IF_ERROR(enc_modular->EncodeStream(get_output(0), aux_out,
                                                  LayerType::ModularGlobal,
                                                  ModularStreamId::Global()));
  }

  std::vector<std::unique_ptr<AuxOut>> aux_outs;
  auto resize_aux_outs = [&aux_outs,
                          aux_out](const size_t num_threads) -> Status {
    if (aux_out == nullptr) {
      aux_outs.resize(num_threads);
    } else {
      while (aux_outs.size() > num_threads) {
        aux_out->Assimilate(*aux_outs.back());
        aux_outs.pop_back();
      }
      while (num_threads > aux_outs.size()) {
        aux_outs.emplace_back(jxl::make_unique<AuxOut>());
      }
    }
    return true;
  };

  const auto process_dc_group = [&](const uint32_t group_index,
                                    const size_t thread) -> Status {
    AuxOut* my_aux_out = aux_outs[thread].get();
    uint32_t input_index = enc_state->streaming_mode ? 0 : group_index;
    BitWriter* output = get_output(input_index + 1);
    if (frame_header.encoding == FrameEncoding::kVarDCT &&
        !(frame_header.flags & FrameHeader::kUseDcFrame)) {
      JXL_RETURN_IF_ERROR(
          output->WithMaxBits(2, LayerType::Dc, my_aux_out, [&] {
            output->Write(2, enc_modular->extra_dc_precision[group_index]);
            return true;
          }));
      JXL_RETURN_IF_ERROR(
          enc_modular->EncodeStream(output, my_aux_out, LayerType::Dc,
                                    ModularStreamId::VarDCTDC(group_index)));
    }
    JXL_RETURN_IF_ERROR(
        enc_modular->EncodeStream(output, my_aux_out, LayerType::ModularDcGroup,
                                  ModularStreamId::ModularDC(group_index)));
    if (frame_header.encoding == FrameEncoding::kVarDCT) {
      const Rect& rect = enc_state->shared.frame_dim.DCGroupRect(input_index);
      size_t nb_bits = CeilLog2Nonzero(rect.xsize() * rect.ysize());
      if (nb_bits != 0) {
        JXL_RETURN_IF_ERROR(output->WithMaxBits(
            nb_bits, LayerType::ControlFields, my_aux_out, [&] {
              output->Write(nb_bits,
                            enc_modular->ac_metadata_size[group_index] - 1);
              return true;
            }));
      }
      JXL_RETURN_IF_ERROR(enc_modular->EncodeStream(
          output, my_aux_out, LayerType::ControlFields,
          ModularStreamId::ACMetadata(group_index)));
    }
    return true;
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, frame_dim.num_dc_groups,
                                resize_aux_outs, process_dc_group,
                                "EncodeDCGroup"));

  if (frame_header.encoding == FrameEncoding::kVarDCT) {
    JXL_RETURN_IF_ERROR(EncodeGlobalACInfo(
        enc_state, get_output(global_ac_index), enc_modular, aux_out));
  }

  const auto process_group = [&](const uint32_t group_index,
                                 const size_t thread) -> Status {
    AuxOut* my_aux_out = aux_outs[thread].get();
    for (size_t i = 0; i < num_passes; i++) {
      if (frame_header.encoding == FrameEncoding::kVarDCT) {
        JXL_RETURN_IF_ERROR(EncodeGroupTokenizedCoefficients(
            group_index, i, enc_state->histogram_idx[group_index], *enc_state,
            ac_group_code(i, group_index), my_aux_out));
      }
      JXL_RETURN_IF_ERROR(enc_modular->EncodeStream(
          ac_group_code(i, group_index), my_aux_out, LayerType::ModularAcGroup,
          ModularStreamId::ModularAC(group_index, i)));
    }
    return true;
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, num_groups, resize_aux_outs,
                                process_group, "EncodeGroupCoefficients"));
  static_cast<void>(resize_aux_outs(0));

  for (std::unique_ptr<BitWriter>& bw : *group_codes) {
    JXL_RETURN_IF_ERROR(bw->WithMaxBits(8, LayerType::Ac, aux_out, [&] {
      bw->ZeroPadToByte();
      return true;
    }));
  }
  return true;
}

Status AppendEncodedFrame(const FrameHeader& frame_header,
                          PassesEncoderState* enc_state,
                          ModularFrameEncoder* enc_modular, ThreadPool* pool,
                          BitWriter* writer, AuxOut* frame_aux_out) {
  std::vector<std::unique_ptr<BitWriter>> group_codes;
  JXL_RETURN_IF_ERROR(EncodeGroups(frame_header, enc_state, enc_modular, pool,
                                   &group_codes, frame_aux_out));
  writer->ZeroPadToByte();
  JXL_RETURN_IF_ERROR(WriteFrameHeader(frame_header, writer, frame_aux_out));
  std::vector<coeff_order_t> permutation;
  JXL_RETURN_IF_ERROR(WriteTocPermutation(permutation, writer, frame_aux_out));
  std::vector<size_t> group_sizes(group_codes.size());
  for (size_t i = 0; i < group_codes.size(); ++i) {
    JXL_ENSURE(group_codes[i]->BitsWritten() % kBitsPerByte == 0);
    group_sizes[i] = group_codes[i]->BitsWritten() / kBitsPerByte;
  }
  JXL_RETURN_IF_ERROR(WriteTocSizes(group_sizes, writer, frame_aux_out));
  JXL_RETURN_IF_ERROR(writer->AppendByteAligned(group_codes));
  return true;
}

void PopulateStatsHeaderFields(const AuxOut& aux, size_t file_bytes,
                               JxlStateHeader* hdr) {
  hdr->total_bits = aux.TotalBits();
  hdr->file_bytes = file_bytes;
  for (size_t i = 0; i < 15; ++i) {
    hdr->layer_bits[i] = aux.layers[i].total_bits;
  }
  hdr->header_bits = hdr->layer_bits[static_cast<size_t>(LayerType::Header)] +
                     hdr->layer_bits[static_cast<size_t>(LayerType::Toc)] +
                     hdr->layer_bits[static_cast<size_t>(LayerType::Quant)] +
                     hdr->layer_bits[static_cast<size_t>(LayerType::Dictionary)] +
                     hdr->layer_bits[static_cast<size_t>(LayerType::Splines)] +
                     hdr->layer_bits[static_cast<size_t>(LayerType::Noise)];
  hdr->dc_bits = hdr->layer_bits[static_cast<size_t>(LayerType::Dc)] +
                 hdr->layer_bits[static_cast<size_t>(LayerType::ModularDcGroup)] +
                 hdr->layer_bits[static_cast<size_t>(LayerType::ModularGlobal)] +
                 hdr->layer_bits[static_cast<size_t>(LayerType::ModularTree)];
  hdr->ac_meta_bits =
      hdr->layer_bits[static_cast<size_t>(LayerType::ControlFields)] +
      hdr->layer_bits[static_cast<size_t>(LayerType::Order)] +
      hdr->layer_bits[static_cast<size_t>(LayerType::Ac)];
  hdr->ac_token_bits =
      hdr->layer_bits[static_cast<size_t>(LayerType::AcTokens)] +
      hdr->layer_bits[static_cast<size_t>(LayerType::ModularAcGroup)];
}

std::string FormatStatsJson(const AuxOut& total_aux, size_t file_bytes,
                            const std::vector<AuxOut>* frame_auxs = nullptr) {
  JxlStateHeader tmp{};
  PopulateStatsHeaderFields(total_aux, file_bytes, &tmp);
  std::ostringstream oss;
  oss << "{\n";
  oss << "  \"file_bytes\": " << tmp.file_bytes << ",\n";
  oss << "  \"total_bytes\": " << tmp.file_bytes << ",\n";
  oss << "  \"total_bits\": " << tmp.total_bits << ",\n";
  oss << "  \"header_bits\": " << tmp.header_bits << ",\n";
  oss << "  \"dc_bits\": " << tmp.dc_bits << ",\n";
  oss << "  \"ac_meta_bits\": " << tmp.ac_meta_bits << ",\n";
  oss << "  \"ac_token_bits\": " << tmp.ac_token_bits << ",\n";
  oss << "  \"layer_bits\": {\n";
  for (size_t i = 0; i < 15; ++i) {
    oss << "    \"" << kLayerNames[i] << "\": " << tmp.layer_bits[i]
        << (i + 1 < 15 ? ",\n" : "\n");
  }
  oss << "  }";
  if (frame_auxs != nullptr && !frame_auxs->empty()) {
    oss << ",\n  \"frames\": [\n";
    for (size_t f = 0; f < frame_auxs->size(); ++f) {
      JxlStateHeader ftmp{};
      PopulateStatsHeaderFields((*frame_auxs)[f], 0, &ftmp);
      oss << "    {\n";
      oss << "      \"frame_index\": " << f << ",\n";
      oss << "      \"total_bits\": " << ftmp.total_bits << ",\n";
      oss << "      \"dc_bits\": " << ftmp.dc_bits << ",\n";
      oss << "      \"ac_meta_bits\": " << ftmp.ac_meta_bits << ",\n";
      oss << "      \"ac_token_bits\": " << ftmp.ac_token_bits << ",\n";
      oss << "      \"layer_bits\": {\n";
      for (size_t i = 0; i < 15; ++i) {
        oss << "        \"" << kLayerNames[i] << "\": " << ftmp.layer_bits[i]
            << (i + 1 < 15 ? ",\n" : "\n");
      }
      oss << "      }\n";
      oss << "    }" << (f + 1 < frame_auxs->size() ? ",\n" : "\n");
    }
    oss << "  ]\n";
  } else {
    oss << "\n";
  }
  oss << "}\n";
  return oss.str();
}

bool WriteTextFile(const std::string& path, const std::string& content) {
  std::ofstream ofs(path, std::ios::out | std::ios::binary | std::ios::trunc);
  if (!ofs) return false;
  ofs.write(content.data(), static_cast<std::streamsize>(content.size()));
  return ofs.good();
}

bool ParseNoiseLut(const std::string& s, NoiseParams* noise_params) {
  std::stringstream ss(s);
  std::string item;
  std::vector<float> vals;
  while (std::getline(ss, item, ',')) {
    char* endptr = nullptr;
    float v = std::strtof(item.c_str(), &endptr);
    if (endptr == item.c_str()) return false;
    vals.push_back(std::max(0.0f, std::min(kNoiseLutMax, v)));
  }
  if (vals.size() != NoiseParams::kNumNoisePoints) return false;
  for (size_t i = 0; i < NoiseParams::kNumNoisePoints; ++i) {
    noise_params->lut[i] = vals[i];
  }
  return true;
}

Status ReadStateFile(const std::string& path, LoadedState* out) {
  std::vector<uint8_t> blob;
  if (!jpegxl::tools::ReadFile(path, &blob)) {
    return JXL_FAILURE("Failed to read state file: %s", path.c_str());
  }
  if (blob.size() < sizeof(JxlStateHeader)) {
    return JXL_FAILURE("State file too small: %s", path.c_str());
  }
  memcpy(&out->hdr, blob.data(), sizeof(JxlStateHeader));
  if (memcmp(out->hdr.magic, "JXLWST01", 8) != 0 || out->hdr.version != 1) {
    return JXL_FAILURE("Invalid state header magic or version in %s",
                       path.c_str());
  }
  const size_t h8 = out->hdr.h8;
  const size_t w8 = out->hdr.w8;
  const size_t ht = out->hdr.ht;
  const size_t wt = out->hdr.wt;
  const size_t n_blocks = h8 * w8;
  const size_t n_tiles = ht * wt;
  const size_t n_dc = 3 * n_blocks;
  const size_t n_ac = 3 * (h8 * 8) * (w8 * 8);

  const size_t needed = sizeof(JxlStateHeader) + n_blocks * sizeof(uint8_t) +
                        n_blocks * sizeof(int32_t) +
                        n_blocks * sizeof(uint8_t) + n_tiles * sizeof(int8_t) +
                        n_tiles * sizeof(int8_t) + n_dc * sizeof(int32_t) +
                        n_ac * sizeof(int32_t);
  if (blob.size() < needed) {
    return JXL_FAILURE("Truncated state payload in %s (got %zu, need %zu)",
                       path.c_str(), blob.size(), needed);
  }

  size_t offset = sizeof(JxlStateHeader);
  auto read_vec = [&](void* dst, size_t bytes) {
    memcpy(dst, blob.data() + offset, bytes);
    offset += bytes;
  };
  out->ac_strategy.resize(n_blocks);
  read_vec(out->ac_strategy.data(), n_blocks * sizeof(uint8_t));

  out->raw_quant_field.resize(n_blocks);
  read_vec(out->raw_quant_field.data(), n_blocks * sizeof(int32_t));

  out->epf_sharpness.resize(n_blocks);
  read_vec(out->epf_sharpness.data(), n_blocks * sizeof(uint8_t));

  out->ytox_map.resize(n_tiles);
  read_vec(out->ytox_map.data(), n_tiles * sizeof(int8_t));

  out->ytob_map.resize(n_tiles);
  read_vec(out->ytob_map.data(), n_tiles * sizeof(int8_t));

  out->dc_quant.resize(n_dc);
  read_vec(out->dc_quant.data(), n_dc * sizeof(int32_t));

  out->quant_ac.resize(n_ac);
  read_vec(out->quant_ac.data(), n_ac * sizeof(int32_t));

  out->has_custom_dequant = false;
  out->force_custom_dequant = (out->hdr.has_dequant_tables & 4u) != 0;
  out->has_custom_dc_quant = false;
  for (size_t c = 0; c < 3; ++c) {
    out->custom_dc_quant[c] = kDCQuant[c];
  }
  for (size_t s = 0; s < AcStrategy::kNumValidStrategies; ++s) {
    out->custom_dequant_matrices[s].clear();
  }

  if ((out->hdr.has_dequant_tables & 1u) != 0) {
    size_t dequant_floats = 0;
    for (size_t s = 0; s < AcStrategy::kNumValidStrategies; ++s) {
      if ((out->hdr.used_acs >> s) & 1u) {
        dequant_floats +=
            3 * 64 * kAcsCoveredBlocksX[s] * kAcsCoveredBlocksY[s];
      }
    }
    if (offset + dequant_floats * sizeof(float) <= blob.size()) {
      for (size_t s = 0; s < AcStrategy::kNumValidStrategies; ++s) {
        if ((out->hdr.used_acs >> s) & 1u) {
          const size_t nf =
              3 * 64 * kAcsCoveredBlocksX[s] * kAcsCoveredBlocksY[s];
          out->custom_dequant_matrices[s].resize(nf);
          read_vec(out->custom_dequant_matrices[s].data(), nf * sizeof(float));
        }
      }
      out->has_custom_dequant = true;
    } else if (offset < blob.size()) {
      return JXL_FAILURE(
          "Truncated custom dequant_matrices in %s (got %zu, need %zu)",
          path.c_str(), blob.size(), offset + dequant_floats * sizeof(float));
    }
  }

  if ((out->hdr.has_dequant_tables & 2u) != 0) {
    if (offset + 3 * sizeof(float) <= blob.size()) {
      read_vec(out->custom_dc_quant, 3 * sizeof(float));
      out->has_custom_dc_quant = true;
    } else {
      return JXL_FAILURE(
          "Truncated custom dc_quant in %s (got %zu, need %zu)", path.c_str(),
          blob.size(), offset + 3 * sizeof(float));
    }
  }

  return true;
}

void ConfigureFrameHeaderLoopFilterFromState(const JxlStateHeader& hdr,
                                             FrameHeader* frame_header) {
  frame_header->x_qm_scale = hdr.x_qm_scale;
  frame_header->b_qm_scale = hdr.b_qm_scale;
  LoopFilter& lf = frame_header->loop_filter;
  lf.gab = (hdr.gab != 0);
  lf.gab_custom = false;
  const float def_w1 = 1.1f * 0.104699568f;
  const float def_w2 = 1.1f * 0.055680538f;
  if (lf.gab) {
    if (std::abs(hdr.gab_weights[0] - def_w1) > 1e-5f ||
        std::abs(hdr.gab_weights[1] - def_w2) > 1e-5f ||
        std::abs(hdr.gab_weights[2] - def_w1) > 1e-5f ||
        std::abs(hdr.gab_weights[3] - def_w2) > 1e-5f ||
        std::abs(hdr.gab_weights[4] - def_w1) > 1e-5f ||
        std::abs(hdr.gab_weights[5] - def_w2) > 1e-5f) {
      lf.gab_custom = true;
      lf.gab_x_weight1 = hdr.gab_weights[0];
      lf.gab_x_weight2 = hdr.gab_weights[1];
      lf.gab_y_weight1 = hdr.gab_weights[2];
      lf.gab_y_weight2 = hdr.gab_weights[3];
      lf.gab_b_weight1 = hdr.gab_weights[4];
      lf.gab_b_weight2 = hdr.gab_weights[5];
    }
  }
  lf.epf_iters = hdr.epf_iters;
  if (lf.epf_iters > 0) {
    if (std::abs(hdr.epf_quant_mul - 0.46f) > 1e-4f ||
        std::abs(hdr.epf_pass0_sigma_scale - 0.9f) > 1e-4f ||
        std::abs(hdr.epf_pass2_sigma_scale - 6.5f) > 1e-4f ||
        std::abs(hdr.epf_border_sad_mul - (2.0f / 3.0f)) > 1e-4f) {
      lf.epf_sigma_custom = true;
      lf.epf_quant_mul = hdr.epf_quant_mul;
      lf.epf_pass0_sigma_scale = hdr.epf_pass0_sigma_scale;
      lf.epf_pass2_sigma_scale = hdr.epf_pass2_sigma_scale;
      lf.epf_border_sad_mul = hdr.epf_border_sad_mul;
    }
    if (std::abs(hdr.epf_channel_scale[0] - 40.0f) > 1e-4f ||
        std::abs(hdr.epf_channel_scale[1] - 5.0f) > 1e-4f ||
        std::abs(hdr.epf_channel_scale[2] - 3.5f) > 1e-4f) {
      lf.epf_weight_custom = true;
      lf.epf_channel_scale[0] = hdr.epf_channel_scale[0];
      lf.epf_channel_scale[1] = hdr.epf_channel_scale[1];
      lf.epf_channel_scale[2] = hdr.epf_channel_scale[2];
    }
  }
}

float QuantMult(float v) {
  if (v > 0.0f) return 1.0f + v;
  return 1.0f / (1.0f - v);
}

float InvQuantMult(float ratio) {
  float r = std::max(1e-6f, ratio);
  if (r >= 1.0f) return r - 1.0f;
  return 1.0f - 1.0f / r;
}

void FitDctQuantParams(size_t wrows, size_t wcols, size_t xs, size_t ys,
                       const float* custom_mat,
                       DctQuantWeightParams* out_params) {
  constexpr size_t B = 8;
  out_params->num_distance_bands = B;
  const float scale =
      static_cast<float>(B - 1) / (1.4142135623730951f + 1e-6f);
  const float rcpcol = scale / static_cast<float>(wcols - 1);
  const float rcprow = scale / static_cast<float>(wrows - 1);
  const size_t plane_size = wrows * wcols;

  for (size_t c = 0; c < 3; ++c) {
    const float* m_plane = custom_mat + c * plane_size;
    double ata[B][B] = {{0.0}};
    double atb[B] = {0.0};
    double mean_log_w = 0.0;
    size_t count = 0;
    for (size_t y = 0; y < wrows; ++y) {
      for (size_t x = 0; x < wcols; ++x) {
        if (y < ys && x < xs) continue;
        float m_val = std::max(1e-12f, m_plane[y * wcols + x]);
        double log_w = -std::log(static_cast<double>(m_val));
        mean_log_w += log_w;
        count++;
        float dy = static_cast<float>(y) * rcprow;
        float dx = static_cast<float>(x) * rcpcol;
        float dist = std::sqrt(dx * dx + dy * dy);
        int idx = static_cast<int>(dist);
        if (idx < 0) idx = 0;
        if (idx >= static_cast<int>(B) - 1) idx = static_cast<int>(B) - 2;
        double frac = static_cast<double>(dist - static_cast<float>(idx));
        double w0 = 1.0 - frac;
        double w1 = frac;
        ata[idx][idx] += w0 * w0;
        ata[idx][idx + 1] += w0 * w1;
        ata[idx + 1][idx] += w1 * w0;
        ata[idx + 1][idx + 1] += w1 * w1;
        atb[idx] += w0 * log_w;
        atb[idx + 1] += w1 * log_w;
      }
    }
    mean_log_w = (count > 0) ? (mean_log_w / static_cast<double>(count)) : 6.0;
    // Regularize slightly toward smooth bands and prior mean.
    for (size_t k = 0; k < B; ++k) {
      ata[k][k] += 1e-6;
      atb[k] += 1e-6 * mean_log_w;
      if (k + 1 < B) {
        ata[k][k] += 1e-4;
        ata[k + 1][k + 1] += 1e-4;
        ata[k][k + 1] -= 1e-4;
        ata[k + 1][k] -= 1e-4;
      }
    }
    // Gaussian elimination with partial pivoting on B x B.
    double z[B] = {0.0};
    for (size_t i = 0; i < B; ++i) {
      size_t pivot = i;
      for (size_t r = i + 1; r < B; ++r) {
        if (std::abs(ata[r][i]) > std::abs(ata[pivot][i])) pivot = r;
      }
      if (pivot != i) {
        for (size_t k = i; k < B; ++k) std::swap(ata[i][k], ata[pivot][k]);
        std::swap(atb[i], atb[pivot]);
      }
      double diag = ata[i][i];
      if (std::abs(diag) < 1e-12) diag = 1e-12;
      for (size_t r = i + 1; r < B; ++r) {
        double factor = ata[r][i] / diag;
        for (size_t k = i; k < B; ++k) ata[r][k] -= factor * ata[i][k];
        atb[r] -= factor * atb[i];
      }
    }
    for (int i = static_cast<int>(B) - 1; i >= 0; --i) {
      double sum = atb[i];
      for (size_t k = static_cast<size_t>(i) + 1; k < B; ++k) {
        sum -= ata[i][k] * z[k];
      }
      double diag = ata[i][i];
      if (std::abs(diag) < 1e-12) diag = 1e-12;
      z[i] = sum / diag;
    }
    float bands[B];
    for (size_t k = 0; k < B; ++k) {
      double clamped_z = std::max(-10.0, std::min(20.0, z[k]));
      bands[k] = static_cast<float>(std::exp(clamped_z));
    }
    out_params->distance_bands[c][0] = std::max(1e-4f, bands[0]);
    for (size_t k = 1; k < B; ++k) {
      out_params->distance_bands[c][k] =
          InvQuantMult(bands[k] / std::max(1e-6f, bands[k - 1]));
    }
  }
}

float EvaluateDctParamsMaxRelErr(size_t wrows, size_t wcols, size_t xs,
                                 size_t ys, const float* custom_mat,
                                 const DctQuantWeightParams& params) {
  const size_t B = params.num_distance_bands;
  const float scale =
      static_cast<float>(B - 1) / (1.4142135623730951f + 1e-6f);
  const float rcpcol = scale / static_cast<float>(wcols - 1);
  const float rcprow = scale / static_cast<float>(wrows - 1);
  const size_t plane_size = wrows * wcols;
  float max_rel_err = 0.0f;
  for (size_t c = 0; c < 3; ++c) {
    float bands[DctQuantWeightParams::kMaxDistanceBands] = {
        params.distance_bands[c][0]};
    for (size_t k = 1; k < B; ++k) {
      bands[k] = bands[k - 1] * QuantMult(params.distance_bands[c][k]);
    }
    const float* m_plane = custom_mat + c * plane_size;
    for (size_t y = 0; y < wrows; ++y) {
      for (size_t x = 0; x < wcols; ++x) {
        if (y < ys && x < xs) continue;
        float dy = static_cast<float>(y) * rcprow;
        float dx = static_cast<float>(x) * rcpcol;
        float dist = std::sqrt(dx * dx + dy * dy);
        int idx = static_cast<int>(dist);
        if (idx < 0) idx = 0;
        if (idx >= static_cast<int>(B) - 1) idx = static_cast<int>(B) - 2;
        float frac = dist - static_cast<float>(idx);
        float w = (B == 1) ? bands[0]
                           : bands[idx] *
                                 std::pow(bands[idx + 1] / bands[idx], frac);
        float m_pred = 1.0f / std::max(1e-12f, w);
        float m_true = std::max(1e-12f, m_plane[y * wcols + x]);
        float rel_err = std::abs(m_pred - m_true) / m_true;
        max_rel_err = std::max(max_rel_err, rel_err);
      }
    }
  }
  return max_rel_err;
}

bool BuildCustomQuantEncodings(const LoadedState& state,
                               const DequantMatrices& default_matrices,
                               std::vector<QuantEncoding>* out_encodings) {
  out_encodings->assign(kNumQuantTables, QuantEncoding::Library<0>());
  const QuantEncoding* library = DequantMatrices::Library();
  bool any_custom = false;

  uint32_t active_qt_mask = 0;
  for (uint8_t s_raw : state.ac_strategy) {
    if (s_raw < AcStrategy::kNumValidStrategies) {
      active_qt_mask |=
          (1u << static_cast<uint32_t>(kAcStrategyToQuantTableMap[s_raw]));
    }
  }
  if (active_qt_mask == 0) {
    active_qt_mask = 1u;
  }

  for (size_t qt_idx = 0; qt_idx < kNumQuantTables; ++qt_idx) {
    if ((active_qt_mask & (1u << qt_idx)) == 0) continue;
    int match_s = -1;
    for (size_t s = 0; s < AcStrategy::kNumValidStrategies; ++s) {
      if (static_cast<size_t>(kAcStrategyToQuantTableMap[s]) == qt_idx &&
          !state.custom_dequant_matrices[s].empty()) {
        match_s = static_cast<int>(s);
        break;
      }
    }
    if (match_s < 0) continue;

    const size_t wrows = 8 * DequantMatrices::required_size_x[qt_idx];
    const size_t wcols = 8 * DequantMatrices::required_size_y[qt_idx];
    const size_t plane_size = wrows * wcols;
    size_t xs = DequantMatrices::required_size_x[qt_idx];
    size_t ys = DequantMatrices::required_size_y[qt_idx];
    CoefficientLayout(&ys, &xs);

    const float* custom_mat = state.custom_dequant_matrices[match_s].data();
    const AcStrategyType acs_type = static_cast<AcStrategyType>(match_s);

    float max_diff_from_def = 0.0f;
    float alpha[3] = {1.0f, 1.0f, 1.0f};
    float max_scaled_err = 0.0f;

    for (size_t c = 0; c < 3; ++c) {
      const float* def_plane = default_matrices.Matrix(acs_type, c);
      const float* cust_plane = custom_mat + c * plane_size;
      double ratio_sum = 0.0;
      size_t count = 0;
      for (size_t y = 0; y < wrows; ++y) {
        for (size_t x = 0; x < wcols; ++x) {
          if (y < ys && x < xs) continue;
          float d_val = std::max(1e-12f, def_plane[y * wcols + x]);
          float c_val = std::max(1e-12f, cust_plane[y * wcols + x]);
          max_diff_from_def =
              std::max(max_diff_from_def, std::abs(c_val - d_val) / d_val);
          ratio_sum += static_cast<double>(c_val) / static_cast<double>(d_val);
          count++;
        }
      }
      alpha[c] = (count > 0)
                     ? static_cast<float>(ratio_sum / static_cast<double>(count))
                     : 1.0f;
      alpha[c] = std::max(1e-4f, std::min(1e4f, alpha[c]));
      for (size_t y = 0; y < wrows; ++y) {
        for (size_t x = 0; x < wcols; ++x) {
          if (y < ys && x < xs) continue;
          float d_val = std::max(1e-12f, def_plane[y * wcols + x]);
          float c_val = std::max(1e-12f, cust_plane[y * wcols + x]);
          float pred = alpha[c] * d_val;
          max_scaled_err =
              std::max(max_scaled_err, std::abs(c_val - pred) / c_val);
        }
      }
    }

    if (max_diff_from_def <= 1e-5f) {
      continue;
    }

    any_custom = true;
    if (max_scaled_err <= 2e-3f) {
      QuantEncoding enc = library[qt_idx];
      for (size_t c = 0; c < 3; ++c) {
        const float inv_alpha = 1.0f / alpha[c];
        switch (enc.mode) {
          case QuantEncoding::kQuantModeDCT:
          case QuantEncoding::kQuantModeDCT4:
          case QuantEncoding::kQuantModeDCT4X8:
            enc.dct_params.distance_bands[c][0] *= inv_alpha;
            break;
          case QuantEncoding::kQuantModeID:
            for (size_t k = 0; k < 3; ++k) enc.idweights[c][k] *= inv_alpha;
            break;
          case QuantEncoding::kQuantModeDCT2:
            for (size_t k = 0; k < 6; ++k) enc.dct2weights[c][k] *= inv_alpha;
            break;
          case QuantEncoding::kQuantModeAFV:
            enc.dct_params.distance_bands[c][0] *= inv_alpha;
            enc.dct_params_afv_4x4.distance_bands[c][0] *= inv_alpha;
            for (size_t k = 0; k < 6; ++k) enc.afv_weights[c][k] *= inv_alpha;
            break;
          default:
            break;
        }
      }
      (*out_encodings)[qt_idx] = enc;
      continue;
    }

    DctQuantWeightParams dct_params{};
    FitDctQuantParams(wrows, wcols, xs, ys, custom_mat, &dct_params);
    float dct_err =
        EvaluateDctParamsMaxRelErr(wrows, wcols, xs, ys, custom_mat, dct_params);
    if (dct_err <= 0.05f) {
      (*out_encodings)[qt_idx] = QuantEncoding::DCT(dct_params);
    } else {
      // Fallback to kQuantModeRAW with exact normal F16 denominator and i16 clamp.
      const float qtable_den = 1.0f / 16384.0f;  // 2^-14, exact normal F16
      std::vector<int> qtable(3 * plane_size, 1);
      for (size_t i = 0; i < 3 * plane_size; ++i) {
        int q = static_cast<int>(
            std::lround(std::max(1e-7f, custom_mat[i]) / qtable_den));
        qtable[i] = std::max(1, std::min(32767, q));
      }
      QuantEncoding raw_enc = QuantEncoding::RAW(std::move(qtable), 0);
      raw_enc.qraw.qtable_den = qtable_den;
      (*out_encodings)[qt_idx] = std::move(raw_enc);
    }
  }

  if (state.force_custom_dequant && !any_custom) {
    (*out_encodings)[0] = library[0];
    any_custom = true;
  }
  return any_custom;
}

Status PopulateEncoderStateFromLoaded(
    JxlMemoryManager* memory_manager, const LoadedState& state, int effort,
    const CodecMetadata* metadata, const FrameHeader& frame_header,
    const std::vector<ImageF>& extra_channels, ThreadPool* pool,
    PassesEncoderState* enc_state,
    std::unique_ptr<ModularFrameEncoder>* out_enc_modular, AuxOut* aux_out) {
  PassesSharedState& shared = enc_state->shared;
  shared.metadata = metadata;
  shared.frame_dim = frame_header.ToFrameDimensions();
  const FrameDimensions& frame_dim = shared.frame_dim;
  const size_t h8 = state.hdr.h8;
  const size_t w8 = state.hdr.w8;
  const size_t ht = state.hdr.ht;
  const size_t wt = state.hdr.wt;

  JXL_ENSURE(frame_dim.xsize == state.hdr.width &&
             frame_dim.ysize == state.hdr.height);
  JXL_ENSURE(frame_dim.xsize_blocks == w8 && frame_dim.ysize_blocks == h8);

  enc_state->cparams.butteraugli_distance = 1.0f;
  int clamped_effort = std::max(1, std::min(9, effort));
  enc_state->cparams.speed_tier = static_cast<SpeedTier>(10 - clamped_effort);
  enc_state->cparams.color_transform = ColorTransform::kXYB;

  shared.image_features.patches.SetShared(&shared.reference_frames);
  JXL_ASSIGN_OR_RETURN(
      shared.ac_strategy,
      AcStrategyImage::Create(memory_manager, frame_dim.xsize_blocks,
                              frame_dim.ysize_blocks));
  shared.ac_strategy.FillInvalid();
  for (size_t by = 0; by < h8; ++by) {
    for (size_t bx = 0; bx < w8; ++bx) {
      if (!shared.ac_strategy.IsValid(bx, by)) {
        uint8_t s = state.ac_strategy[by * w8 + bx];
        JXL_ENSURE(AcStrategy::IsRawStrategyValid(s));
        AcStrategy acs = AcStrategy::FromRawStrategy(s);
        const size_t cx = acs.covered_blocks_x();
        const size_t cy = acs.covered_blocks_y();
        JXL_ENSURE(bx + cx <= w8 && by + cy <= h8);
        JXL_ENSURE((bx / kGroupDimInBlocks) ==
                   ((bx + cx - 1) / kGroupDimInBlocks));
        JXL_ENSURE((by / kGroupDimInBlocks) ==
                   ((by + cy - 1) / kGroupDimInBlocks));
        JXL_RETURN_IF_ERROR(shared.ac_strategy.SetNoBoundsCheck(
            bx, by, static_cast<AcStrategyType>(s), /*check=*/true));
      }
    }
  }

  JXL_ASSIGN_OR_RETURN(shared.raw_quant_field,
                       ImageI::Create(memory_manager, w8, h8));
  JXL_ASSIGN_OR_RETURN(shared.epf_sharpness,
                       ImageB::Create(memory_manager, w8, h8));
  double rqf_sum = 0.0;
  for (size_t by = 0; by < h8; ++by) {
    int32_t* q_row = shared.raw_quant_field.Row(by);
    uint8_t* s_row = shared.epf_sharpness.Row(by);
    for (size_t bx = 0; bx < w8; ++bx) {
      q_row[bx] = std::max<int32_t>(
          1, std::min<int32_t>(256, state.raw_quant_field[by * w8 + bx]));
      s_row[bx] = std::min<uint8_t>(7, state.epf_sharpness[by * w8 + bx]);
      rqf_sum += static_cast<double>(q_row[bx]);
    }
  }
  const float mean_rqf = static_cast<float>(
      rqf_sum / std::max<size_t>(1, h8 * w8));
  const float est_distance = std::max(
      0.1f,
      std::min(15.0f, (0.79f * 65536.0f) /
                          std::max(1.0f, static_cast<float>(state.hdr.global_scale) *
                                             mean_rqf)));
  enc_state->cparams.butteraugli_distance = est_distance;

  JXL_ASSIGN_OR_RETURN(shared.cmap,
                       ColorCorrelationMap::Create(
                           memory_manager, frame_dim.xsize, frame_dim.ysize));
  {
    ColorCorrelation cc;
    if (state.hdr.color_factor > 0) {
      cc.SetColorFactor(static_cast<uint32_t>(state.hdr.color_factor));
    }
    cc.SetYToXDC(std::max<int32_t>(-128, std::min<int32_t>(127, state.hdr.ytox_dc)));
    cc.SetYToBDC(std::max<int32_t>(-128, std::min<int32_t>(127, state.hdr.ytob_dc)));
    BitWriter cc_bw(memory_manager);
    JXL_RETURN_IF_ERROR(
        ColorCorrelationEncodeDC(cc, &cc_bw, LayerType::Dc, nullptr));
    cc_bw.ZeroPadToByte();
    BitReader cc_br(cc_bw.GetSpan());
    JXL_RETURN_IF_ERROR(shared.cmap.DecodeDC(&cc_br));
    JXL_RETURN_IF_ERROR(cc_br.Close());
  }
  for (size_t ty = 0; ty < ht; ++ty) {
    int8_t* x_row = shared.cmap.ytox_map.Row(ty);
    int8_t* b_row = shared.cmap.ytob_map.Row(ty);
    for (size_t tx = 0; tx < wt; ++tx) {
      x_row[tx] = state.ytox_map[ty * wt + tx];
      b_row[tx] = state.ytob_map[ty * wt + tx];
    }
  }

  shared.coeff_order_size = kCoeffOrderMaxSize;
  shared.coeff_orders.resize(frame_header.passes.num_passes *
                             kCoeffOrderMaxSize);
  JXL_ASSIGN_OR_RETURN(shared.quant_dc, ImageB::Create(memory_manager, w8, h8));
  JXL_ASSIGN_OR_RETURN(shared.dc_storage,
                       Image3F::Create(memory_manager, w8, h8));
  shared.dc = &shared.dc_storage;

  enc_state->x_qm_multiplier =
      std::pow(1.25f, frame_header.x_qm_scale - 2.0f);
  enc_state->b_qm_multiplier =
      std::pow(1.25f, frame_header.b_qm_scale - 2.0f);

  enc_state->passes.resize(1);
  enc_state->passes[0].ac_tokens.resize(frame_dim.num_groups);
  enc_state->coeffs.clear();
  JXL_ASSIGN_OR_RETURN(
      std::unique_ptr<ACImageT<int32_t>> coeffs,
      ACImageT<int32_t>::Make(memory_manager, kGroupDim * kGroupDim,
                              frame_dim.num_groups));
  enc_state->coeffs.emplace_back(std::move(coeffs));

  // Copy quant_ac (3, H8*8, W8*8) into enc_state->coeffs[0] group by group.
  const size_t ac_stride = w8 * 8;
  const size_t ac_plane_size = (h8 * 8) * ac_stride;
  for (size_t group_index = 0; group_index < frame_dim.num_groups;
       ++group_index) {
    const Rect block_group_rect = frame_dim.BlockGroupRect(group_index);
    int32_t* ac_rows[3] = {
        enc_state->coeffs[0]->PlaneRow(0, group_index, 0).ptr32,
        enc_state->coeffs[0]->PlaneRow(1, group_index, 0).ptr32,
        enc_state->coeffs[0]->PlaneRow(2, group_index, 0).ptr32,
    };
    size_t offset = 0;
    for (size_t by = 0; by < block_group_rect.ysize(); ++by) {
      AcStrategyRow acs_row =
          shared.ac_strategy.ConstRow(block_group_rect, by);
      const size_t g_by = block_group_rect.y0() + by;
      for (size_t bx = 0; bx < block_group_rect.xsize(); ++bx) {
        AcStrategy acs = acs_row[bx];
        if (!acs.IsFirstBlock()) continue;
        const size_t g_bx = block_group_rect.x0() + bx;
        const size_t cx = acs.covered_blocks_x();
        const size_t cy = acs.covered_blocks_y();
        const size_t size = kDCTBlockSize * cx * cy;
        size_t cx_layout = cx;
        size_t cy_layout = cy;
        CoefficientLayout(&cy_layout, &cx_layout);
        for (size_t c = 0; c < 3; ++c) {
          const int32_t* src_plane = state.quant_ac.data() + c * ac_plane_size;
          int32_t* dst = ac_rows[c] + offset;
          for (size_t r = 0; r < 8 * cy; ++r) {
            const int32_t* src_row =
                src_plane + (g_by * 8 + r) * ac_stride + (g_bx * 8);
            memcpy(dst + r * (8 * cx), src_row, (8 * cx) * sizeof(int32_t));
          }
          for (size_t iy = 0; iy < cy_layout; ++iy) {
            for (size_t ix = 0; ix < cx_layout; ++ix) {
              dst[iy * (8 * cx_layout) + ix] = 0;
            }
          }
        }
        offset += size;
      }
    }
  }

  // Construct ModularFrameEncoder BEFORE DequantMatricesSetCustom so any
  // kQuantModeRAW tables can register their Modular streams via AddQuantTable.
  CompressParams mod_cparams = enc_state->cparams;
  mod_cparams.modular_mode = false;
  if (!extra_channels.empty()) {
    mod_cparams.butteraugli_distance = 0.0f;
    mod_cparams.ec_distance.assign(metadata->m.num_extra_channels, 0.0f);
    // Work around upstream libjxl ModularFrameDecoder::MaybeDropFullImage()
    // (lib/jxl/dec_modular.cc:278,321), which drops full_image when a single
    // extra channel has shift >= 3 (ecups >= 8 * frame_upsampling), exceeds
    // group_dim (256), and has full_image.transform.empty(). Enabling lossless
    // Squeeze (responsive = 1) populates full_image.transform so unmodified
    // djxl preserves full_image and decodes exact extra-channel values.
    bool need_squeeze_for_shift3_ec = false;
    for (size_t ec = 0; ec < frame_header.extra_channel_upsampling.size();
         ++ec) {
      const uint32_t ecups = frame_header.extra_channel_upsampling[ec];
      if (ecups >= 8 * frame_header.upsampling) {
        const size_t ec_w = DivCeil(frame_dim.xsize_upsampled, ecups);
        const size_t ec_h = DivCeil(frame_dim.ysize_upsampled, ecups);
        if (ec_w > frame_dim.group_dim || ec_h > frame_dim.group_dim) {
          need_squeeze_for_shift3_ec = true;
          break;
        }
      }
    }
    mod_cparams.responsive = need_squeeze_for_shift3_ec ? 1 : 0;
  }
  JXL_ASSIGN_OR_RETURN(
      *out_enc_modular,
      ModularFrameEncoder::Create(memory_manager, frame_header, mod_cparams,
                                  /*streaming_mode=*/false));
  ModularFrameEncoder* enc_modular = out_enc_modular->get();

  const uint32_t all_acs_mask = (1u << AcStrategy::kNumValidStrategies) - 1u;
  JXL_RETURN_IF_ERROR(
      shared.matrices.EnsureComputed(memory_manager, all_acs_mask));
  if (state.has_custom_dequant || state.force_custom_dequant) {
    std::vector<QuantEncoding> custom_encodings;
    if (BuildCustomQuantEncodings(state, shared.matrices, &custom_encodings)) {
      JXL_RETURN_IF_ERROR(DequantMatricesSetCustom(
          &shared.matrices, custom_encodings, enc_modular));
      JXL_RETURN_IF_ERROR(
          shared.matrices.EnsureComputed(memory_manager, all_acs_mask));
    }
  }

  bool custom_dc = false;
  if (state.has_custom_dc_quant) {
    for (size_t c = 0; c < 3; ++c) {
      if (std::abs(state.custom_dc_quant[c] - kDCQuant[c]) > 1e-7f) {
        custom_dc = true;
      }
    }
  }
  if (custom_dc) {
    const float inv_dc[3] = {
        1.0f / std::max(1e-12f, state.custom_dc_quant[0]),
        1.0f / std::max(1e-12f, state.custom_dc_quant[1]),
        1.0f / std::max(1e-12f, state.custom_dc_quant[2]),
    };
    JXL_RETURN_IF_ERROR(
        DequantMatricesSetCustomDC(memory_manager, &shared.matrices, inv_dc));
  } else {
    JXL_RETURN_IF_ERROR(
        DequantMatricesScaleDC(memory_manager, &shared.matrices, 1.0f));
  }

  shared.quantizer =
      Quantizer(shared.matrices, state.hdr.quant_dc, state.hdr.global_scale);
  shared.quantizer.RecomputeFromGlobalScale();

  if (!extra_channels.empty()) {
    Rect group_rect(0, 0, frame_dim.xsize_padded, frame_dim.ysize_padded);
    Rect frame_area_rect(0, 0, frame_dim.xsize, frame_dim.ysize);
    JXL_RETURN_IF_ERROR(enc_modular->ComputeEncodingData(
        frame_header, metadata->m, /*color=*/nullptr, extra_channels,
        group_rect, frame_dim, frame_area_rect, enc_state, *JxlGetDefaultCms(),
        pool, aux_out, /*do_color=*/false));
  }

  // Reconstruct float DC image from integer dc_quant (0:X, 1:Y, 2:B) so
  // AddVarDCTDC(..., /*nl_dc=*/false, ...) recovers exact dc_quant integers.
  JXL_ASSIGN_OR_RETURN(Image3F dc, Image3F::Create(memory_manager, w8, h8));
  const float dc_step[3] = {
      shared.quantizer.GetDcStep(0),
      shared.quantizer.GetDcStep(1),
      shared.quantizer.GetDcStep(2),
  };
  const float* cfl_dc = shared.cmap.base().DCFactors();
  const size_t dc_plane_size = h8 * w8;
  for (size_t by = 0; by < h8; ++by) {
    float* row_x = dc.PlaneRow(0, by);
    float* row_y = dc.PlaneRow(1, by);
    float* row_b = dc.PlaneRow(2, by);
    for (size_t bx = 0; bx < w8; ++bx) {
      const size_t idx = by * w8 + bx;
      const int32_t q_x = state.dc_quant[0 * dc_plane_size + idx];
      const int32_t q_y = state.dc_quant[1 * dc_plane_size + idx];
      const int32_t q_b = state.dc_quant[2 * dc_plane_size + idx];
      const float flt_y = static_cast<float>(q_y) * dc_step[1];
      row_y[bx] = flt_y;
      row_x[bx] = static_cast<float>(q_x) * dc_step[0] + flt_y * cfl_dc[0];
      row_b[bx] = static_cast<float>(q_b) * dc_step[2] + flt_y * cfl_dc[2];
    }
  }

  for (size_t group_index = 0; group_index < frame_dim.num_dc_groups;
       ++group_index) {
    const Rect r = frame_dim.DCGroupRect(group_index);
    JXL_RETURN_IF_ERROR(enc_modular->AddVarDCTDC(
        frame_header, dc, r, group_index, /*nl_dc=*/false, enc_state,
        /*jpeg_transcode=*/false));
  }

  {
    CompressParams ctx_cparams = enc_state->cparams;
    if (ctx_cparams.speed_tier >= SpeedTier::kFalcon) {
      ctx_cparams.speed_tier = SpeedTier::kSquirrel;
    }
    FindBestBlockEntropyModel(ctx_cparams, shared.raw_quant_field,
                              shared.ac_strategy, &shared.block_ctx_map);
  }

  JXL_RETURN_IF_ERROR(ComputeACMetadata(pool, enc_state, enc_modular));
  enc_state->used_acs = 0;
  JXL_RETURN_IF_ERROR(ComputeAllCoeffOrders(*enc_state, frame_dim));
  shared.num_histograms = 1;
  enc_state->histogram_idx.assign(frame_dim.num_groups, 0);
  JXL_RETURN_IF_ERROR(
      TokenizeAllCoefficients(frame_header, pool, enc_state));

  JXL_RETURN_IF_ERROR(enc_modular->ComputeTree(pool));
  JXL_RETURN_IF_ERROR(enc_modular->ComputeTokens(pool));
  return true;
}

Status RunDumpInit(const std::string& input_path, float distance, int effort,
                   const std::string& output_state_path,
                   const std::string& output_jxl_path,
                   const std::string& json_out_path) {
  JxlMemoryManager* memory_manager = jpegxl::tools::NoMemoryManager();
  jpegxl::tools::ThreadPoolInternal pool(4);

  std::vector<uint8_t> encoded;
  if (!jpegxl::tools::ReadFile(input_path, &encoded)) {
    return JXL_FAILURE("Failed to read input image: %s", input_path.c_str());
  }
  jxl::extras::PackedPixelFile ppf;
  if (!jxl::extras::DecodeBytes(Bytes(encoded), jxl::extras::ColorHints(),
                                &ppf)) {
    return JXL_FAILURE("Failed to decode input image: %s", input_path.c_str());
  }
  auto io = jxl::make_unique<CodecInOut>(memory_manager);
  JXL_RETURN_IF_ERROR(jxl::extras::ConvertPackedPixelFileToCodecInOut(
      ppf, pool.get(), io.get()));

  CompressParams cparams;
  cparams.butteraugli_distance = distance;
  cparams.original_butteraugli_distance = distance;
  int clamped_effort = std::max(1, std::min(9, effort));
  cparams.speed_tier = static_cast<SpeedTier>(10 - clamped_effort);
  cparams.color_transform = ColorTransform::kXYB;
  cparams.patches = Override::kOff;
  cparams.dots = Override::kOff;
  cparams.noise = Override::kOff;
  cparams.progressive_dc = 0;
  cparams.resampling = 1;
  cparams.ec_resampling = 1;

  CodecMetadata metadata;
  JXL_RETURN_IF_ERROR(metadata.size.Set(io->xsize(), io->ysize()));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB(/*is_gray=*/false);
  metadata.m.xyb_encoded = true;

  ProgressiveSplitter progressive_splitter;
  FrameHeader frame_header(&metadata);
  frame_header.frame_type = FrameType::kRegularFrame;
  frame_header.encoding = FrameEncoding::kVarDCT;
  frame_header.color_transform = ColorTransform::kXYB;
  frame_header.is_last = true;
  JXL_RETURN_IF_ERROR(progressive_splitter.InitPasses(&frame_header.passes));
  JXL_RETURN_IF_ERROR(LoopFilterFromParams(cparams, &frame_header));

  PassesEncoderState enc_state(memory_manager);
  enc_state.cparams = cparams;
  PassesSharedState& shared = enc_state.shared;
  shared.metadata = &metadata;
  shared.frame_dim = frame_header.ToFrameDimensions();
  const FrameDimensions& frame_dim = shared.frame_dim;
  shared.image_features.patches.SetShared(&shared.reference_frames);

  JXL_ASSIGN_OR_RETURN(
      shared.ac_strategy,
      AcStrategyImage::Create(memory_manager, frame_dim.xsize_blocks,
                              frame_dim.ysize_blocks));
  JXL_ASSIGN_OR_RETURN(shared.raw_quant_field,
                       ImageI::Create(memory_manager, frame_dim.xsize_blocks,
                                      frame_dim.ysize_blocks));
  JXL_ASSIGN_OR_RETURN(shared.epf_sharpness,
                       ImageB::Create(memory_manager, frame_dim.xsize_blocks,
                                      frame_dim.ysize_blocks));
  JXL_ASSIGN_OR_RETURN(
      shared.cmap, ColorCorrelationMap::Create(memory_manager, frame_dim.xsize,
                                               frame_dim.ysize));
  shared.coeff_order_size = kCoeffOrderMaxSize;
  shared.coeff_orders.resize(frame_header.passes.num_passes *
                             kCoeffOrderMaxSize);
  JXL_ASSIGN_OR_RETURN(shared.quant_dc,
                       ImageB::Create(memory_manager, frame_dim.xsize_blocks,
                                      frame_dim.ysize_blocks));
  JXL_ASSIGN_OR_RETURN(shared.dc_storage,
                       Image3F::Create(memory_manager, frame_dim.xsize_blocks,
                                       frame_dim.ysize_blocks));
  shared.dc = &shared.dc_storage;

  JXL_ASSIGN_OR_RETURN(
      Image3F opsin,
      Image3F::Create(memory_manager, RoundUpToBlockDim(frame_dim.xsize),
                      RoundUpToBlockDim(frame_dim.ysize)));
  JXL_RETURN_IF_ERROR(opsin.ShrinkTo(frame_dim.xsize, frame_dim.ysize));
  JXL_RETURN_IF_ERROR(CopyImageTo(*io->Main().color(), &opsin));

  Image3F linear_storage;
  Image3F* linear = nullptr;
  if (cparams.speed_tier <= SpeedTier::kKitten) {
    JXL_ASSIGN_OR_RETURN(
        linear_storage,
        Image3F::Create(memory_manager, frame_dim.xsize, frame_dim.ysize));
    linear = &linear_storage;
  }
  JXL_RETURN_IF_ERROR(ToXYB(io->Main().c_current(),
                            metadata.m.IntensityTarget(), nullptr, pool.get(),
                            &opsin, *JxlGetDefaultCms(), linear));
  JXL_RETURN_IF_ERROR(PadImageToBlockMultipleInPlace(&opsin));

  Rect group_rect(opsin);
  ComputeChromacityAdjustments(cparams, opsin, group_rect, &frame_header);

  JXL_ASSIGN_OR_RETURN(
      std::unique_ptr<ModularFrameEncoder> enc_modular,
      ModularFrameEncoder::Create(memory_manager, frame_header, cparams,
                                  /*streaming_mode=*/false));

  enc_state.passes.resize(1);
  enc_state.passes[0].ac_tokens.resize(frame_dim.num_groups);

  Image3F orig_opsin;
  JXL_ASSIGN_OR_RETURN(
      orig_opsin,
      Image3F::Create(memory_manager, group_rect.xsize(), group_rect.ysize()));
  JXL_RETURN_IF_ERROR(
      CopyImageTo(group_rect, opsin, Rect(orig_opsin), &orig_opsin));
  JXL_RETURN_IF_ERROR(orig_opsin.ShrinkTo(frame_dim.xsize, frame_dim.ysize));

  AuxOut heuristics_aux;
  JXL_RETURN_IF_ERROR(LossyFrameHeuristics(
      frame_header, &enc_state, enc_modular.get(), linear, &opsin, group_rect,
      *JxlGetDefaultCms(), pool.get(), &heuristics_aux));

  const uint32_t all_acs_mask = (1u << AcStrategy::kNumValidStrategies) - 1u;
  JXL_RETURN_IF_ERROR(
      shared.matrices.EnsureComputed(memory_manager, all_acs_mask));

  enc_state.x_qm_multiplier =
      std::pow(1.25f, frame_header.x_qm_scale - 2.0f);
  enc_state.b_qm_multiplier =
      std::pow(1.25f, frame_header.b_qm_scale - 2.0f);

  JXL_ASSIGN_OR_RETURN(
      std::unique_ptr<ACImageT<int32_t>> coeffs,
      ACImageT<int32_t>::Make(memory_manager, kGroupDim * kGroupDim,
                              frame_dim.num_groups));
  enc_state.coeffs.clear();
  enc_state.coeffs.emplace_back(std::move(coeffs));

  float scale =
      shared.quantizer.ScaleGlobalScale(enc_state.cparams.quant_ac_rescale);
  JXL_RETURN_IF_ERROR(
      DequantMatricesScaleDC(memory_manager, &shared.matrices, scale));
  shared.quantizer.RecomputeFromGlobalScale();

  JXL_ASSIGN_OR_RETURN(
      Image3F dc, Image3F::Create(memory_manager, frame_dim.xsize_blocks,
                                  frame_dim.ysize_blocks));
  for (size_t group_idx = 0; group_idx < frame_dim.num_groups; ++group_idx) {
    JXL_RETURN_IF_ERROR(
        ComputeCoefficients(group_idx, &enc_state, opsin, group_rect, &dc));
  }

  for (size_t group_index = 0; group_index < frame_dim.num_dc_groups;
       ++group_index) {
    const Rect r = frame_dim.DCGroupRect(group_index);
    JXL_RETURN_IF_ERROR(enc_modular->AddVarDCTDC(
        frame_header, dc, r, group_index, /*nl_dc=*/false, &enc_state,
        /*jpeg_transcode=*/false));
  }

  // Extract exact quantized DC integers (0:X, 1:Y, 2:B) matching nl_dc=false.
  const size_t h8 = frame_dim.ysize_blocks;
  const size_t w8 = frame_dim.xsize_blocks;
  const size_t ht = DivCeil(h8, kColorTileDimInBlocks);
  const size_t wt = DivCeil(w8, kColorTileDimInBlocks);
  std::vector<int32_t> dc_quant_vec(3 * h8 * w8, 0);
  {
    const float inv_dc_step[3] = {
        shared.quantizer.GetInvDcStep(0),
        shared.quantizer.GetInvDcStep(1),
        shared.quantizer.GetInvDcStep(2),
    };
    const float y_dc_step = shared.quantizer.GetDcStep(1);
    const float* cfl_dc = shared.cmap.base().DCFactors();
    const size_t plane_sz = h8 * w8;
    for (size_t by = 0; by < h8; ++by) {
      const float* row_x = dc.ConstPlaneRow(0, by);
      const float* row_y = dc.ConstPlaneRow(1, by);
      const float* row_b = dc.ConstPlaneRow(2, by);
      for (size_t bx = 0; bx < w8; ++bx) {
        const size_t idx = by * w8 + bx;
        const int32_t q_y =
            static_cast<int32_t>(std::round(row_y[bx] * inv_dc_step[1]));
        const int32_t q_x = static_cast<int32_t>(std::round(
            (row_x[bx] - q_y * (y_dc_step * cfl_dc[0])) * inv_dc_step[0]));
        const int32_t q_b = static_cast<int32_t>(std::round(
            (row_b[bx] - q_y * (y_dc_step * cfl_dc[2])) * inv_dc_step[2]));
        dc_quant_vec[0 * plane_sz + idx] = q_x;
        dc_quant_vec[1 * plane_sz + idx] = q_y;
        dc_quant_vec[2 * plane_sz + idx] = q_b;
      }
    }
  }

  if (!(frame_header.flags & FrameHeader::kSkipAdaptiveDCSmoothing)) {
    JXL_RETURN_IF_ERROR(AdaptiveDCSmoothing(
        memory_manager, shared.quantizer.MulDC(), &shared.dc_storage,
        pool.get()));
  }
  JXL_RETURN_IF_ERROR(ComputeARHeuristics(frame_header, &enc_state, orig_opsin,
                                          group_rect, pool.get()));
  JXL_RETURN_IF_ERROR(
      ComputeACMetadata(pool.get(), &enc_state, enc_modular.get()));

  enc_state.used_acs = 0;
  JXL_RETURN_IF_ERROR(ComputeAllCoeffOrders(enc_state, frame_dim));
  shared.num_histograms = 1;
  enc_state.histogram_idx.assign(frame_dim.num_groups, 0);
  JXL_RETURN_IF_ERROR(
      TokenizeAllCoefficients(frame_header, pool.get(), &enc_state));
  JXL_RETURN_IF_ERROR(enc_modular->ComputeTree(pool.get()));
  JXL_RETURN_IF_ERROR(enc_modular->ComputeTokens(pool.get()));

  BitWriter writer(memory_manager);
  AuxOut total_aux;
  JXL_RETURN_IF_ERROR(WriteCodestreamHeaders(&metadata, &writer, &total_aux));
  JXL_RETURN_IF_ERROR(AppendEncodedFrame(frame_header, &enc_state,
                                         enc_modular.get(), pool.get(), &writer,
                                         &total_aux));
  Span<const uint8_t> jxl_bytes = writer.GetSpan();
  if (!output_jxl_path.empty()) {
    if (!jpegxl::tools::WriteFile(output_jxl_path, jxl_bytes)) {
      return JXL_FAILURE("Failed to write output JXL: %s",
                         output_jxl_path.c_str());
    }
  }

  // Build JxlStateHeader and serialize <state.bin>
  JxlStateHeader hdr{};
  memcpy(hdr.magic, "JXLWST01", 8);
  hdr.version = 1;
  hdr.height = static_cast<uint32_t>(frame_dim.ysize);
  hdr.width = static_cast<uint32_t>(frame_dim.xsize);
  hdr.h8 = static_cast<uint32_t>(h8);
  hdr.w8 = static_cast<uint32_t>(w8);
  hdr.ht = static_cast<uint32_t>(ht);
  hdr.wt = static_cast<uint32_t>(wt);
  QuantizerParams qparams = shared.quantizer.GetParams();
  hdr.global_scale = static_cast<int32_t>(qparams.global_scale);
  hdr.quant_dc = static_cast<int32_t>(qparams.quant_dc);
  hdr.x_qm_scale = static_cast<int32_t>(frame_header.x_qm_scale);
  hdr.b_qm_scale = static_cast<int32_t>(frame_header.b_qm_scale);
  hdr.color_factor =
      static_cast<int32_t>(shared.cmap.base().GetColorFactor());
  hdr.base_correlation_x = shared.cmap.base().GetBaseCorrelationX();
  hdr.base_correlation_b = shared.cmap.base().GetBaseCorrelationB();
  hdr.ytox_dc = shared.cmap.base().GetYToXDC();
  hdr.ytob_dc = shared.cmap.base().GetYToBDC();
  const LoopFilter& lf = frame_header.loop_filter;
  hdr.epf_iters = static_cast<int32_t>(lf.epf_iters);
  hdr.epf_quant_mul = lf.epf_quant_mul;
  hdr.epf_pass0_sigma_scale = lf.epf_pass0_sigma_scale;
  hdr.epf_pass2_sigma_scale = lf.epf_pass2_sigma_scale;
  hdr.epf_border_sad_mul = lf.epf_border_sad_mul;
  for (size_t c = 0; c < 3; ++c) {
    hdr.epf_channel_scale[c] = lf.epf_channel_scale[c];
  }
  for (size_t i = 0; i < 8; ++i) {
    hdr.epf_sharp_lut[i] = lf.epf_sharp_lut[i];
  }
  hdr.gab = lf.gab ? 1u : 0u;
  hdr.gab_weights[0] = lf.gab_x_weight1;
  hdr.gab_weights[1] = lf.gab_x_weight2;
  hdr.gab_weights[2] = lf.gab_y_weight1;
  hdr.gab_weights[3] = lf.gab_y_weight2;
  hdr.gab_weights[4] = lf.gab_b_weight1;
  hdr.gab_weights[5] = lf.gab_b_weight2;
  hdr.used_acs = all_acs_mask;
  hdr.has_dequant_tables = 1u;
  PopulateStatsHeaderFields(total_aux, jxl_bytes.size(), &hdr);

  std::vector<uint8_t> ac_strategy_vec(h8 * w8, 0);
  std::vector<int32_t> raw_quant_vec(h8 * w8, 1);
  std::vector<uint8_t> epf_sharp_vec(h8 * w8, 0);
  for (size_t by = 0; by < h8; ++by) {
    AcStrategyRow acs_row = shared.ac_strategy.ConstRow(by);
    const int32_t* q_row = shared.raw_quant_field.ConstRow(by);
    const uint8_t* s_row = shared.epf_sharpness.ConstRow(by);
    for (size_t bx = 0; bx < w8; ++bx) {
      ac_strategy_vec[by * w8 + bx] = acs_row[bx].RawStrategy();
      raw_quant_vec[by * w8 + bx] = q_row[bx];
      epf_sharp_vec[by * w8 + bx] = s_row[bx];
    }
  }

  std::vector<int8_t> ytox_vec(ht * wt, 0);
  std::vector<int8_t> ytob_vec(ht * wt, 0);
  for (size_t ty = 0; ty < ht; ++ty) {
    const int8_t* x_row = shared.cmap.ytox_map.ConstRow(ty);
    const int8_t* b_row = shared.cmap.ytob_map.ConstRow(ty);
    for (size_t tx = 0; tx < wt; ++tx) {
      ytox_vec[ty * wt + tx] = x_row[tx];
      ytob_vec[ty * wt + tx] = b_row[tx];
    }
  }

  const size_t ac_stride = w8 * 8;
  const size_t ac_plane_size = (h8 * 8) * ac_stride;
  std::vector<int32_t> quant_ac_vec(3 * ac_plane_size, 0);
  for (size_t group_index = 0; group_index < frame_dim.num_groups;
       ++group_index) {
    const Rect block_group_rect = frame_dim.BlockGroupRect(group_index);
    const int32_t* ac_rows[3] = {
        enc_state.coeffs[0]->PlaneRow(0, group_index, 0).ptr32,
        enc_state.coeffs[0]->PlaneRow(1, group_index, 0).ptr32,
        enc_state.coeffs[0]->PlaneRow(2, group_index, 0).ptr32,
    };
    size_t offset = 0;
    for (size_t by = 0; by < block_group_rect.ysize(); ++by) {
      AcStrategyRow acs_row =
          shared.ac_strategy.ConstRow(block_group_rect, by);
      const size_t g_by = block_group_rect.y0() + by;
      for (size_t bx = 0; bx < block_group_rect.xsize(); ++bx) {
        AcStrategy acs = acs_row[bx];
        if (!acs.IsFirstBlock()) continue;
        const size_t g_bx = block_group_rect.x0() + bx;
        const size_t cx = acs.covered_blocks_x();
        const size_t cy = acs.covered_blocks_y();
        const size_t size = kDCTBlockSize * cx * cy;
        for (size_t c = 0; c < 3; ++c) {
          int32_t* dst_plane = quant_ac_vec.data() + c * ac_plane_size;
          const int32_t* src = ac_rows[c] + offset;
          for (size_t r = 0; r < 8 * cy; ++r) {
            int32_t* dst_row =
                dst_plane + (g_by * 8 + r) * ac_stride + (g_bx * 8);
            memcpy(dst_row, src + r * (8 * cx), (8 * cx) * sizeof(int32_t));
          }
        }
        offset += size;
      }
    }
  }

  std::ofstream ofs(output_state_path,
                    std::ios::out | std::ios::binary | std::ios::trunc);
  if (!ofs) {
    return JXL_FAILURE("Failed to open output state file: %s",
                       output_state_path.c_str());
  }
  ofs.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
  ofs.write(reinterpret_cast<const char*>(ac_strategy_vec.data()),
            ac_strategy_vec.size() * sizeof(uint8_t));
  ofs.write(reinterpret_cast<const char*>(raw_quant_vec.data()),
            raw_quant_vec.size() * sizeof(int32_t));
  ofs.write(reinterpret_cast<const char*>(epf_sharp_vec.data()),
            epf_sharp_vec.size() * sizeof(uint8_t));
  ofs.write(reinterpret_cast<const char*>(ytox_vec.data()),
            ytox_vec.size() * sizeof(int8_t));
  ofs.write(reinterpret_cast<const char*>(ytob_vec.data()),
            ytob_vec.size() * sizeof(int8_t));
  ofs.write(reinterpret_cast<const char*>(dc_quant_vec.data()),
            dc_quant_vec.size() * sizeof(int32_t));
  ofs.write(reinterpret_cast<const char*>(quant_ac_vec.data()),
            quant_ac_vec.size() * sizeof(int32_t));

  for (size_t s = 0; s < AcStrategy::kNumValidStrategies; ++s) {
    if ((all_acs_mask >> s) & 1u) {
      const size_t n_floats =
          kDCTBlockSize * kAcsCoveredBlocksX[s] * kAcsCoveredBlocksY[s];
      for (size_t c = 0; c < 3; ++c) {
        const float* mat =
            shared.matrices.Matrix(static_cast<AcStrategyType>(s), c);
        ofs.write(reinterpret_cast<const char*>(mat),
                  n_floats * sizeof(float));
      }
    }
  }
  if (!ofs.good()) {
    return JXL_FAILURE("Error writing state file: %s",
                       output_state_path.c_str());
  }

  std::string stats_json = FormatStatsJson(total_aux, jxl_bytes.size());
  if (!json_out_path.empty()) {
    if (!WriteTextFile(json_out_path, stats_json)) {
      return JXL_FAILURE("Failed to write JSON stats: %s",
                         json_out_path.c_str());
    }
  } else {
    std::cout << stats_json;
  }
  return true;
}

Status RunEncodeSingle(const std::string& state_path,
                       const std::string& output_jxl_path, int effort,
                       const std::string& noise_lut_str,
                       const std::string& json_out_path) {
  JxlMemoryManager* memory_manager = jpegxl::tools::NoMemoryManager();
  jpegxl::tools::ThreadPoolInternal pool(4);

  LoadedState state;
  JXL_RETURN_IF_ERROR(ReadStateFile(state_path, &state));

  CodecMetadata metadata;
  JXL_RETURN_IF_ERROR(metadata.size.Set(state.hdr.width, state.hdr.height));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB(/*is_gray=*/false);
  metadata.m.xyb_encoded = true;

  ProgressiveSplitter progressive_splitter;
  FrameHeader frame_header(&metadata);
  frame_header.frame_type = FrameType::kRegularFrame;
  frame_header.encoding = FrameEncoding::kVarDCT;
  frame_header.color_transform = ColorTransform::kXYB;
  frame_header.upsampling = 1;
  frame_header.save_before_color_transform = false;
  frame_header.custom_size_or_origin = false;
  frame_header.is_last = true;
  frame_header.flags = 0;
  JXL_RETURN_IF_ERROR(progressive_splitter.InitPasses(&frame_header.passes));
  ConfigureFrameHeaderLoopFilterFromState(state.hdr, &frame_header);

  PassesEncoderState enc_state(memory_manager);
  if (!noise_lut_str.empty()) {
    if (!ParseNoiseLut(noise_lut_str,
                       &enc_state.shared.image_features.noise_params)) {
      return JXL_FAILURE("Invalid --noise-lut argument: %s",
                         noise_lut_str.c_str());
    }
    frame_header.UpdateFlag(true, FrameHeader::kNoise);
  }

  std::vector<ImageF> empty_ec;
  std::unique_ptr<ModularFrameEncoder> enc_modular;
  AuxOut total_aux;
  JXL_RETURN_IF_ERROR(PopulateEncoderStateFromLoaded(
      memory_manager, state, effort, &metadata, frame_header, empty_ec,
      pool.get(), &enc_state, &enc_modular, &total_aux));

  BitWriter writer(memory_manager);
  JXL_RETURN_IF_ERROR(WriteCodestreamHeaders(&metadata, &writer, &total_aux));
  JXL_RETURN_IF_ERROR(AppendEncodedFrame(frame_header, &enc_state,
                                         enc_modular.get(), pool.get(), &writer,
                                         &total_aux));

  Span<const uint8_t> jxl_bytes = writer.GetSpan();
  if (!jpegxl::tools::WriteFile(output_jxl_path, jxl_bytes)) {
    return JXL_FAILURE("Failed to write output JXL: %s",
                       output_jxl_path.c_str());
  }

  std::string stats_json = FormatStatsJson(total_aux, jxl_bytes.size());
  if (!json_out_path.empty()) {
    if (!WriteTextFile(json_out_path, stats_json)) {
      return JXL_FAILURE("Failed to write JSON stats: %s",
                         json_out_path.c_str());
    }
  } else {
    std::cout << stats_json;
  }
  return true;
}

Status LoadPatchesFile(const std::string& patches_path,
                       std::vector<JxlPatchEntry>* out_patches) {
  std::vector<uint8_t> blob;
  if (!jpegxl::tools::ReadFile(patches_path, &blob)) {
    return JXL_FAILURE("Failed to read patches file: %s",
                       patches_path.c_str());
  }
  out_patches->clear();
  if (blob.size() >= sizeof(JxlPatchesHeader) &&
      memcmp(blob.data(), "JXLWPT01", 8) == 0) {
    JxlPatchesHeader phdr{};
    memcpy(&phdr, blob.data(), sizeof(JxlPatchesHeader));
    if (phdr.version != 1) {
      return JXL_FAILURE("Unsupported patches version: %u", phdr.version);
    }
    const size_t needed =
        sizeof(JxlPatchesHeader) + phdr.num_patches * sizeof(JxlPatchEntry);
    if (blob.size() < needed) {
      return JXL_FAILURE("Truncated binary patches file: %s",
                         patches_path.c_str());
    }
    out_patches->resize(phdr.num_patches);
    if (phdr.num_patches > 0) {
      memcpy(out_patches->data(), blob.data() + sizeof(JxlPatchesHeader),
             phdr.num_patches * sizeof(JxlPatchEntry));
    }
    return true;
  }

  // Fallback: text format "<ref> <x0> <y0> <xsize> <ysize> <x> <y> <mode>"
  std::string text(reinterpret_cast<const char*>(blob.data()), blob.size());
  std::istringstream iss(text);
  std::string line;
  while (std::getline(iss, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream lss(line);
    JxlPatchEntry p{};
    if (lss >> p.ref >> p.x0 >> p.y0 >> p.xsize >> p.ysize >> p.x >> p.y >>
        p.blend_mode) {
      out_patches->push_back(p);
    }
  }
  return true;
}

Status RunEncodePatches(const std::string& ref_state_path,
                        const std::string& main_state_path,
                        uint32_t upsampling, const std::string& patches_path,
                        const std::string& output_jxl_path, int effort,
                        const std::string& noise_lut_str,
                        const std::string& json_out_path) {
  if (upsampling != 2 && upsampling != 4 && upsampling != 8) {
    return JXL_FAILURE("--upsampling must be 2, 4, or 8 (got %u)", upsampling);
  }
  JxlMemoryManager* memory_manager = jpegxl::tools::NoMemoryManager();
  jpegxl::tools::ThreadPoolInternal pool(4);

  LoadedState ref_state;
  LoadedState main_state;
  JXL_RETURN_IF_ERROR(ReadStateFile(ref_state_path, &ref_state));
  JXL_RETURN_IF_ERROR(ReadStateFile(main_state_path, &main_state));

  std::vector<JxlPatchEntry> patch_entries;
  if (!patches_path.empty()) {
    JXL_RETURN_IF_ERROR(LoadPatchesFile(patches_path, &patch_entries));
  } else {
    patch_entries.push_back(JxlPatchEntry{
        /*ref=*/0,
        /*x0=*/0,
        /*y0=*/0,
        /*xsize=*/main_state.hdr.width,
        /*ysize=*/main_state.hdr.height,
        /*x=*/0,
        /*y=*/0,
        /*blend_mode=*/static_cast<uint32_t>(PatchBlendMode::kAdd),
    });
  }
  if (patch_entries.size() != 1) {
    return JXL_FAILURE(
        "--encode-patches requires exactly 1 full-frame patch (got %zu)",
        patch_entries.size());
  }
  {
    const auto& p = patch_entries[0];
    if (p.ref != 0 || p.x0 != 0 || p.y0 != 0 || p.x != 0 || p.y != 0 ||
        p.xsize != main_state.hdr.width || p.ysize != main_state.hdr.height ||
        p.blend_mode != static_cast<uint32_t>(PatchBlendMode::kAdd)) {
      return JXL_FAILURE(
          "--encode-patches requires a single full-frame (0, 0, %u, %u) kAdd "
          "(2) patch (got ref=%u, (%u,%u)->(%u,%u) %ux%u, mode=%u)",
          main_state.hdr.width, main_state.hdr.height, p.ref, p.x0, p.y0, p.x,
          p.y, p.xsize, p.ysize, p.blend_mode);
    }
  }

  CodecMetadata metadata;
  JXL_RETURN_IF_ERROR(
      metadata.size.Set(main_state.hdr.width, main_state.hdr.height));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB(/*is_gray=*/false);
  metadata.m.xyb_encoded = true;

  BitWriter writer(memory_manager);
  AuxOut total_aux;
  std::vector<AuxOut> frame_auxs(2);
  JXL_RETURN_IF_ERROR(WriteCodestreamHeaders(&metadata, &writer, &total_aux));
  writer.ZeroPadToByte();

  // Frame 0: kReferenceOnly low-res frame in XYB
  {
    ProgressiveSplitter splitter0;
    FrameHeader frame0(&metadata);
    frame0.frame_type = FrameType::kReferenceOnly;
    frame0.encoding = FrameEncoding::kVarDCT;
    frame0.color_transform = ColorTransform::kXYB;
    frame0.upsampling = upsampling;
    frame0.save_as_reference = 0;
    frame0.save_before_color_transform = true;
    frame0.custom_size_or_origin = false;
    frame0.is_last = false;
    frame0.flags = 0;
    JXL_RETURN_IF_ERROR(splitter0.InitPasses(&frame0.passes));
    ConfigureFrameHeaderLoopFilterFromState(ref_state.hdr, &frame0);

    PassesEncoderState enc_state0(memory_manager);
    std::vector<ImageF> empty_ec;
    std::unique_ptr<ModularFrameEncoder> enc_modular0;
    JXL_RETURN_IF_ERROR(PopulateEncoderStateFromLoaded(
        memory_manager, ref_state, effort, &metadata, frame0, empty_ec,
        pool.get(), &enc_state0, &enc_modular0, &frame_auxs[0]));
    JXL_RETURN_IF_ERROR(AppendEncodedFrame(frame0, &enc_state0,
                                           enc_modular0.get(), pool.get(),
                                           &writer, &frame_auxs[0]));
    total_aux.Assimilate(frame_auxs[0]);
  }

  // Frame 1: kRegularFrame high-res frame with PatchDictionary
  {
    ProgressiveSplitter splitter1;
    FrameHeader frame1(&metadata);
    frame1.frame_type = FrameType::kRegularFrame;
    frame1.encoding = FrameEncoding::kVarDCT;
    frame1.color_transform = ColorTransform::kXYB;
    frame1.upsampling = 1;
    frame1.save_before_color_transform = false;
    frame1.custom_size_or_origin = false;
    frame1.is_last = true;
    frame1.flags = 0;
    frame1.UpdateFlag(!patch_entries.empty(), FrameHeader::kPatches);
    JXL_RETURN_IF_ERROR(splitter1.InitPasses(&frame1.passes));
    ConfigureFrameHeaderLoopFilterFromState(main_state.hdr, &frame1);

    PassesEncoderState enc_state1(memory_manager);
    if (!noise_lut_str.empty()) {
      if (!ParseNoiseLut(noise_lut_str,
                         &enc_state1.shared.image_features.noise_params)) {
        return JXL_FAILURE("Invalid --noise-lut argument: %s",
                           noise_lut_str.c_str());
      }
      frame1.UpdateFlag(true, FrameHeader::kNoise);
    }

    std::vector<ImageF> empty_ec;
    std::unique_ptr<ModularFrameEncoder> enc_modular1;
    JXL_RETURN_IF_ERROR(PopulateEncoderStateFromLoaded(
        memory_manager, main_state, effort, &metadata, frame1, empty_ec,
        pool.get(), &enc_state1, &enc_modular1, &frame_auxs[1]));

    if (!patch_entries.empty()) {
      std::vector<PatchReferencePosition> ref_positions;
      std::vector<PatchPosition> positions;
      std::vector<PatchBlending> blendings;
      const size_t num_ec = metadata.m.extra_channel_info.size();
      const size_t blendings_stride = num_ec + 1;

      ref_positions.reserve(patch_entries.size());
      positions.reserve(patch_entries.size());
      blendings.reserve(patch_entries.size() * blendings_stride);

      for (const auto& p : patch_entries) {
        JXL_ENSURE(p.ref == 0);
        JXL_ENSURE(p.xsize >= 1 && p.ysize >= 1);
        JXL_ENSURE(static_cast<uint64_t>(p.x0) + p.xsize <=
                       main_state.hdr.width &&
                   static_cast<uint64_t>(p.y0) + p.ysize <=
                       main_state.hdr.height);
        JXL_ENSURE(static_cast<uint64_t>(p.x) + p.xsize <=
                       main_state.hdr.width &&
                   static_cast<uint64_t>(p.y) + p.ysize <=
                       main_state.hdr.height);
        JXL_ENSURE(p.blend_mode == static_cast<uint32_t>(PatchBlendMode::kAdd));

        size_t ref_idx = ref_positions.size();
        if (!ref_positions.empty() && ref_positions.back().ref == p.ref &&
            ref_positions.back().x0 == p.x0 &&
            ref_positions.back().y0 == p.y0 &&
            ref_positions.back().xsize == p.xsize &&
            ref_positions.back().ysize == p.ysize) {
          ref_idx = ref_positions.size() - 1;
        } else {
          ref_positions.push_back(
              PatchReferencePosition{p.ref, p.x0, p.y0, p.xsize, p.ysize});
        }
        positions.push_back(PatchPosition{p.x, p.y, ref_idx});
        blendings.push_back(PatchBlending{
            static_cast<PatchBlendMode>(p.blend_mode), /*alpha_channel=*/0,
            /*clamp=*/false});
        for (size_t ec = 0; ec < num_ec; ++ec) {
          blendings.push_back(PatchBlending{PatchBlendMode::kNone, 0, false});
        }
      }

      PatchDictionaryEncoder::SetPositions(
          &enc_state1.shared.image_features.patches, std::move(positions),
          std::move(ref_positions), std::move(blendings), blendings_stride);
    }

    JXL_RETURN_IF_ERROR(AppendEncodedFrame(frame1, &enc_state1,
                                           enc_modular1.get(), pool.get(),
                                           &writer, &frame_auxs[1]));
    total_aux.Assimilate(frame_auxs[1]);
  }

  Span<const uint8_t> jxl_bytes = writer.GetSpan();
  if (!jpegxl::tools::WriteFile(output_jxl_path, jxl_bytes)) {
    return JXL_FAILURE("Failed to write output JXL: %s",
                       output_jxl_path.c_str());
  }

  std::string stats_json =
      FormatStatsJson(total_aux, jxl_bytes.size(), &frame_auxs);
  if (!json_out_path.empty()) {
    if (!WriteTextFile(json_out_path, stats_json)) {
      return JXL_FAILURE("Failed to write JSON stats: %s",
                         json_out_path.c_str());
    }
  } else {
    std::cout << stats_json;
  }
  return true;
}

Status LoadAlphaFile(const std::string& alpha_path, size_t expected_h,
                     size_t expected_w, uint32_t expected_dim_shift,
                     std::vector<uint8_t>* out_alpha) {
  std::vector<uint8_t> blob;
  if (!jpegxl::tools::ReadFile(alpha_path, &blob)) {
    return JXL_FAILURE("Failed to read alpha file: %s", alpha_path.c_str());
  }
  const size_t n_pixels = expected_h * expected_w;
  if (blob.size() >= sizeof(JxlAlphaHeader) &&
      memcmp(blob.data(), "JXLWAL01", 8) == 0) {
    JxlAlphaHeader ahdr{};
    memcpy(&ahdr, blob.data(), sizeof(JxlAlphaHeader));
    if (ahdr.version != 1 || ahdr.height != expected_h ||
        ahdr.width != expected_w || ahdr.dim_shift != expected_dim_shift) {
      return JXL_FAILURE(
          "Alpha header mismatch: got (%u, %u, shift=%u), expected (%zu, %zu, "
          "shift=%u)",
          ahdr.height, ahdr.width, ahdr.dim_shift, expected_h, expected_w,
          expected_dim_shift);
    }
    if (blob.size() < sizeof(JxlAlphaHeader) + n_pixels) {
      return JXL_FAILURE("Truncated binary alpha file: %s", alpha_path.c_str());
    }
    out_alpha->assign(blob.begin() + sizeof(JxlAlphaHeader),
                      blob.begin() + sizeof(JxlAlphaHeader) + n_pixels);
    return true;
  }
  if (blob.size() == n_pixels) {
    *out_alpha = std::move(blob);
    return true;
  }
  return JXL_FAILURE("Invalid alpha file size (%zu bytes) for (%zu x %zu)",
                     blob.size(), expected_h, expected_w);
}

uint32_t ParseAlphaBlendModeArg(const std::string& s) {
  if (s == "kBlendAbove")
    return static_cast<uint32_t>(PatchBlendMode::kBlendAbove);
  if (s == "kBlendBelow")
    return static_cast<uint32_t>(PatchBlendMode::kBlendBelow);
  if (s == "kAlphaWeightedAddAbove")
    return static_cast<uint32_t>(PatchBlendMode::kAlphaWeightedAddAbove);
  if (s == "kAlphaWeightedAddBelow")
    return static_cast<uint32_t>(PatchBlendMode::kAlphaWeightedAddBelow);
  char* endptr = nullptr;
  unsigned long val = std::strtoul(s.c_str(), &endptr, 10);
  if (endptr == s.c_str() || *endptr != '\0') {
    return 0;
  }
  return static_cast<uint32_t>(val);
}

Status RunEncodeAlpha(const std::string& ref_state_path,
                      const std::string& main_state_path, uint32_t upsampling,
                      uint32_t dim_shift, const std::string& alpha_path,
                      const std::string& patches_path,
                      uint32_t alpha_blend_mode, uint32_t alpha_bits,
                      bool alpha_clamp, const std::string& output_jxl_path,
                      int effort, const std::string& noise_lut_str,
                      const std::string& json_out_path) {
  if (upsampling != 2 && upsampling != 4 && upsampling != 8) {
    return JXL_FAILURE("--upsampling must be 2, 4, or 8 (got %u)", upsampling);
  }
  if (dim_shift > 3) {
    return JXL_FAILURE("--dim-shift must be in {0, 1, 2, 3} (got %u)",
                       dim_shift);
  }
  const auto is_valid_alpha_bmode = [](uint32_t bm) {
    return bm == static_cast<uint32_t>(PatchBlendMode::kBlendAbove) ||
           bm == static_cast<uint32_t>(PatchBlendMode::kBlendBelow) ||
           bm ==
               static_cast<uint32_t>(PatchBlendMode::kAlphaWeightedAddAbove) ||
           bm == static_cast<uint32_t>(PatchBlendMode::kAlphaWeightedAddBelow);
  };
  if (!is_valid_alpha_bmode(alpha_blend_mode)) {
    return JXL_FAILURE(
        "Unsupported --alpha-blend-mode: %u (must be in {4, 5, 6, 7})",
        alpha_blend_mode);
  }
  JxlMemoryManager* memory_manager = jpegxl::tools::NoMemoryManager();
  jpegxl::tools::ThreadPoolInternal pool(4);

  LoadedState ref_state;
  LoadedState main_state;
  JXL_RETURN_IF_ERROR(ReadStateFile(ref_state_path, &ref_state));
  JXL_RETURN_IF_ERROR(ReadStateFile(main_state_path, &main_state));

  const size_t full_w = main_state.hdr.width;
  const size_t full_h = main_state.hdr.height;
  const uint32_t eff_ec_ups0 = std::max(upsampling, 1u << dim_shift);
  const uint32_t eff_ec_ups1 = 1u << dim_shift;
  const size_t w_alpha = DivCeil(full_w, eff_ec_ups1);
  const size_t h_alpha = DivCeil(full_h, eff_ec_ups1);

  std::vector<uint8_t> alpha_u8;
  JXL_RETURN_IF_ERROR(
      LoadAlphaFile(alpha_path, h_alpha, w_alpha, dim_shift, &alpha_u8));

  std::vector<JxlPatchEntry> patch_entries;
  if (!patches_path.empty()) {
    JXL_RETURN_IF_ERROR(LoadPatchesFile(patches_path, &patch_entries));
  }
  if (patch_entries.empty()) {
    patch_entries.push_back(JxlPatchEntry{
        /*ref=*/0,
        /*x0=*/0,
        /*y0=*/0,
        /*xsize=*/static_cast<uint32_t>(full_w),
        /*ysize=*/static_cast<uint32_t>(full_h),
        /*x=*/0,
        /*y=*/0,
        /*blend_mode=*/alpha_blend_mode,
    });
  }
  if (patch_entries.size() != 1) {
    return JXL_FAILURE(
        "--encode-alpha requires exactly 1 full-frame patch (got %zu)",
        patch_entries.size());
  }
  {
    const auto& p = patch_entries[0];
    const uint32_t bmode =
        (p.blend_mode == 0) ? alpha_blend_mode : p.blend_mode;
    if (p.ref != 0 || p.x0 != 0 || p.y0 != 0 || p.x != 0 || p.y != 0 ||
        p.xsize != full_w || p.ysize != full_h || !is_valid_alpha_bmode(bmode)) {
      return JXL_FAILURE(
          "--encode-alpha requires a single full-frame (0, 0, %zu, %zu) alpha "
          "patch in {4, 5, 6, 7} (got ref=%u, (%u,%u)->(%u,%u) %ux%u, mode=%u)",
          full_w, full_h, p.ref, p.x0, p.y0, p.x, p.y, p.xsize, p.ysize, bmode);
    }
  }

  CodecMetadata metadata;
  JXL_RETURN_IF_ERROR(metadata.size.Set(full_w, full_h));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB(/*is_gray=*/false);
  metadata.m.xyb_encoded = true;

  ExtraChannelInfo eci;
  eci.type = ExtraChannel::kAlpha;
  eci.bit_depth.bits_per_sample = (alpha_bits >= 1 && alpha_bits <= 16)
                                      ? alpha_bits
                                      : 8;
  eci.bit_depth.floating_point_sample = false;
  eci.bit_depth.exponent_bits_per_sample = 0;
  eci.dim_shift = dim_shift;
  eci.alpha_associated = false;
  metadata.m.extra_channel_info.clear();
  metadata.m.extra_channel_info.push_back(eci);
  metadata.m.num_extra_channels = 1;

  BitWriter writer(memory_manager);
  AuxOut total_aux;
  std::vector<AuxOut> frame_auxs(2);
  JXL_RETURN_IF_ERROR(WriteCodestreamHeaders(&metadata, &writer, &total_aux));
  writer.ZeroPadToByte();

  // Frame 0: kReferenceOnly low-res frame in XYB (save_before_color_transform = true)
  // with opaque 1.0f alpha in extra channel 0.
  {
    ProgressiveSplitter splitter0;
    FrameHeader frame0(&metadata);
    frame0.frame_type = FrameType::kReferenceOnly;
    frame0.encoding = FrameEncoding::kVarDCT;
    frame0.color_transform = ColorTransform::kXYB;
    frame0.upsampling = upsampling;
    frame0.extra_channel_upsampling.clear();
    frame0.extra_channel_upsampling.push_back(eff_ec_ups0);
    frame0.save_as_reference = 0;
    frame0.save_before_color_transform = true;
    frame0.custom_size_or_origin = false;
    frame0.is_last = false;
    frame0.flags = 0;
    JXL_RETURN_IF_ERROR(splitter0.InitPasses(&frame0.passes));
    ConfigureFrameHeaderLoopFilterFromState(ref_state.hdr, &frame0);

    FrameDimensions frame0_dim = frame0.ToFrameDimensions();
    const size_t w_ec0 = DivCeil(full_w, eff_ec_ups0);
    const size_t h_ec0 = DivCeil(full_h, eff_ec_ups0);
    JXL_ASSIGN_OR_RETURN(
        ImageF dummy_alpha,
        ImageF::Create(memory_manager,
                       std::max(frame0_dim.xsize_padded, w_ec0),
                       std::max(frame0_dim.ysize_padded, h_ec0)));
    FillImage(1.0f, &dummy_alpha);
    std::vector<ImageF> ec_images0;
    ec_images0.push_back(std::move(dummy_alpha));

    PassesEncoderState enc_state0(memory_manager);
    std::unique_ptr<ModularFrameEncoder> enc_modular0;
    JXL_RETURN_IF_ERROR(PopulateEncoderStateFromLoaded(
        memory_manager, ref_state, effort, &metadata, frame0, ec_images0,
        pool.get(), &enc_state0, &enc_modular0, &frame_auxs[0]));
    JXL_RETURN_IF_ERROR(AppendEncodedFrame(frame0, &enc_state0,
                                           enc_modular0.get(), pool.get(),
                                           &writer, &frame_auxs[0]));
    total_aux.Assimilate(frame_auxs[0]);
  }

  // Frame 1: kRegularFrame with PatchDictionary XYB alpha blending from Frame 0
  {
    ProgressiveSplitter splitter1;
    FrameHeader frame1(&metadata);
    frame1.frame_type = FrameType::kRegularFrame;
    frame1.encoding = FrameEncoding::kVarDCT;
    frame1.color_transform = ColorTransform::kXYB;
    frame1.upsampling = 1;
    frame1.extra_channel_upsampling.clear();
    frame1.extra_channel_upsampling.push_back(eff_ec_ups1);
    frame1.save_before_color_transform = false;
    frame1.custom_size_or_origin = false;
    frame1.is_last = true;
    frame1.flags = 0;
    frame1.UpdateFlag(true, FrameHeader::kPatches);
    JXL_RETURN_IF_ERROR(splitter1.InitPasses(&frame1.passes));
    ConfigureFrameHeaderLoopFilterFromState(main_state.hdr, &frame1);

    // Frame-level blending is kReplace; blending occurs in XYB via PatchDictionary.
    frame1.blending_info.mode = BlendMode::kReplace;
    frame1.blending_info.source = 0;
    frame1.blending_info.alpha_channel = 0;
    frame1.blending_info.clamp = false;

    frame1.extra_channel_blending_info.resize(1);
    frame1.extra_channel_blending_info[0].mode = BlendMode::kReplace;
    frame1.extra_channel_blending_info[0].source = 0;
    frame1.extra_channel_blending_info[0].alpha_channel = 0;
    frame1.extra_channel_blending_info[0].clamp = false;

    PassesEncoderState enc_state1(memory_manager);
    if (!noise_lut_str.empty()) {
      if (!ParseNoiseLut(noise_lut_str,
                         &enc_state1.shared.image_features.noise_params)) {
        return JXL_FAILURE("Invalid --noise-lut argument: %s",
                           noise_lut_str.c_str());
      }
      frame1.UpdateFlag(true, FrameHeader::kNoise);
    }

    FrameDimensions frame1_dim = frame1.ToFrameDimensions();
    JXL_ASSIGN_OR_RETURN(
        ImageF alpha_image,
        ImageF::Create(memory_manager,
                       std::max(frame1_dim.xsize_padded, w_alpha),
                       std::max(frame1_dim.ysize_padded, h_alpha)));
    ZeroFillImage(&alpha_image);
    const float inv_255 = 1.0f / 255.0f;
    for (size_t y = 0; y < h_alpha; ++y) {
      float* JXL_RESTRICT row = alpha_image.Row(y);
      for (size_t x = 0; x < w_alpha; ++x) {
        row[x] = static_cast<float>(alpha_u8[y * w_alpha + x]) * inv_255;
      }
    }
    std::vector<ImageF> ec_images1;
    ec_images1.push_back(std::move(alpha_image));

    std::unique_ptr<ModularFrameEncoder> enc_modular1;
    JXL_RETURN_IF_ERROR(PopulateEncoderStateFromLoaded(
        memory_manager, main_state, effort, &metadata, frame1, ec_images1,
        pool.get(), &enc_state1, &enc_modular1, &frame_auxs[1]));

    // Populate PatchDictionary with color blend mode and EC kReplace to restore
    // opaque 1.0 alpha after XYB color blending.
    std::vector<PatchReferencePosition> ref_positions;
    std::vector<PatchPosition> positions;
    std::vector<PatchBlending> blendings;
    const size_t num_ec = metadata.m.extra_channel_info.size();
    const size_t blendings_stride = num_ec + 1;

    ref_positions.reserve(patch_entries.size());
    positions.reserve(patch_entries.size());
    blendings.reserve(patch_entries.size() * blendings_stride);

    for (const auto& p : patch_entries) {
      JXL_ENSURE(p.ref == 0);
      JXL_ENSURE(p.xsize >= 1 && p.ysize >= 1);
      JXL_ENSURE(static_cast<uint64_t>(p.x0) + p.xsize <= full_w &&
                 static_cast<uint64_t>(p.y0) + p.ysize <= full_h);
      JXL_ENSURE(static_cast<uint64_t>(p.x) + p.xsize <= full_w &&
                 static_cast<uint64_t>(p.y) + p.ysize <= full_h);
      const uint32_t bmode =
          (p.blend_mode == 0) ? alpha_blend_mode : p.blend_mode;
      JXL_ENSURE(is_valid_alpha_bmode(bmode));

      size_t ref_idx = ref_positions.size();
      if (!ref_positions.empty() && ref_positions.back().ref == p.ref &&
          ref_positions.back().x0 == p.x0 && ref_positions.back().y0 == p.y0 &&
          ref_positions.back().xsize == p.xsize &&
          ref_positions.back().ysize == p.ysize) {
        ref_idx = ref_positions.size() - 1;
      } else {
        ref_positions.push_back(
            PatchReferencePosition{p.ref, p.x0, p.y0, p.xsize, p.ysize});
      }
      positions.push_back(PatchPosition{p.x, p.y, ref_idx});
      blendings.push_back(PatchBlending{
          static_cast<PatchBlendMode>(bmode), /*alpha_channel=*/0,
          /*clamp=*/alpha_clamp});
      for (size_t ec = 0; ec < num_ec; ++ec) {
        blendings.push_back(
            PatchBlending{PatchBlendMode::kReplace, /*alpha_channel=*/0,
                          /*clamp=*/false});
      }
    }

    PatchDictionaryEncoder::SetPositions(
        &enc_state1.shared.image_features.patches, std::move(positions),
        std::move(ref_positions), std::move(blendings), blendings_stride);

    JXL_RETURN_IF_ERROR(AppendEncodedFrame(frame1, &enc_state1,
                                           enc_modular1.get(), pool.get(),
                                           &writer, &frame_auxs[1]));
    total_aux.Assimilate(frame_auxs[1]);
  }

  Span<const uint8_t> jxl_bytes = writer.GetSpan();
  if (!jpegxl::tools::WriteFile(output_jxl_path, jxl_bytes)) {
    return JXL_FAILURE("Failed to write output JXL: %s",
                       output_jxl_path.c_str());
  }

  std::string stats_json =
      FormatStatsJson(total_aux, jxl_bytes.size(), &frame_auxs);
  if (!json_out_path.empty()) {
    if (!WriteTextFile(json_out_path, stats_json)) {
      return JXL_FAILURE("Failed to write JSON stats: %s",
                         json_out_path.c_str());
    }
  } else {
    std::cout << stats_json;
  }
  return true;
}

const char* FrameTypeName(FrameType ft) {
  switch (ft) {
    case FrameType::kRegularFrame:
      return "kRegularFrame";
    case FrameType::kDCFrame:
      return "kDCFrame";
    case FrameType::kReferenceOnly:
      return "kReferenceOnly";
    case FrameType::kSkipProgressive:
      return "kSkipProgressive";
  }
  return "kUnknown";
}

const char* BlendModeName(BlendMode bm) {
  switch (bm) {
    case BlendMode::kReplace:
      return "kReplace";
    case BlendMode::kAdd:
      return "kAdd";
    case BlendMode::kBlend:
      return "kBlend";
    case BlendMode::kAlphaWeightedAdd:
      return "kAlphaWeightedAdd";
    case BlendMode::kMul:
      return "kMul";
  }
  return "kUnknown";
}

const char* ExtraChannelTypeName(ExtraChannel ec) {
  switch (ec) {
    case ExtraChannel::kAlpha:
      return "kAlpha";
    case ExtraChannel::kDepth:
      return "kDepth";
    case ExtraChannel::kSpotColor:
      return "kSpotColor";
    case ExtraChannel::kSelectionMask:
      return "kSelectionMask";
    case ExtraChannel::kBlack:
      return "kBlack";
    case ExtraChannel::kCFA:
      return "kCFA";
    case ExtraChannel::kThermal:
      return "kThermal";
    case ExtraChannel::kReserved0:
    case ExtraChannel::kReserved1:
    case ExtraChannel::kReserved2:
    case ExtraChannel::kReserved3:
    case ExtraChannel::kReserved4:
    case ExtraChannel::kReserved5:
    case ExtraChannel::kReserved6:
    case ExtraChannel::kReserved7:
    case ExtraChannel::kUnknown:
    case ExtraChannel::kOptional:
      return "kOther";
  }
  return "kOther";
}

Status RunInspect(const std::string& input_jxl_path,
                  const std::string& json_out_path) {
  JxlMemoryManager* memory_manager = jpegxl::tools::NoMemoryManager();
  jpegxl::tools::ThreadPoolInternal pool(4);

  std::vector<uint8_t> compressed;
  if (!jpegxl::tools::ReadFile(input_jxl_path, &compressed)) {
    return JXL_FAILURE("Failed to read JXL file: %s", input_jxl_path.c_str());
  }

  Bytes compressed_span(compressed);
  BitReader br(compressed_span);
  Status close_status = true;
  BitReaderScopedCloser br_closer(br, close_status);
  if (br.ReadFixedBits<8>() != 0xFF ||
      br.ReadFixedBits<8>() != kCodestreamMarker) {
    return JXL_FAILURE("Invalid JXL codestream signature");
  }
  CodecMetadata metadata;
  JXL_RETURN_IF_ERROR(ReadSizeHeader(&br, &metadata.size));
  JXL_RETURN_IF_ERROR(ReadImageMetadata(&br, &metadata.m));
  metadata.transform_data.nonserialized_xyb_encoded = metadata.m.xyb_encoded;
  JXL_RETURN_IF_ERROR(Bundle::Read(&br, &metadata.transform_data));
  JXL_RETURN_IF_ERROR(br.JumpToByteBoundary());

  size_t pos = br.TotalBitsConsumed() / kBitsPerByte;

  auto dec_state = jxl::make_unique<PassesDecoderState>(memory_manager);
  JXL_RETURN_IF_ERROR(
      dec_state->output_encoding_info.SetFromMetadata(metadata));

  struct FrameInspectInfo {
    size_t index = 0;
    std::string frame_type;
    uint32_t frame_type_id = 0;
    std::string encoding;
    size_t width = 0;
    size_t height = 0;
    uint32_t upsampling = 1;
    std::vector<uint32_t> ec_upsampling;
    std::vector<uint32_t> effective_ec_upsampling;
    bool save_before_color_transform = false;
    uint32_t save_as_reference = 0;
    bool is_last = false;
    bool has_noise = false;
    float noise_lut[8] = {0};
    bool has_patches = false;
    size_t num_patches = 0;
    size_t num_ref_patches = 0;
    std::vector<uint32_t> patch_blend_modes;
    std::vector<uint32_t> patch_ec_blend_modes;
    std::vector<uint32_t> patch_alpha_channels;
    std::vector<bool> patch_clamps;
    std::string blending_mode;
    uint32_t blending_mode_id = 0;
    uint32_t blending_source = 0;
    uint32_t alpha_channel = 0;
    int32_t global_scale = 0;
    int32_t quant_dc = 0;
    uint32_t x_qm_scale = 0;
    uint32_t b_qm_scale = 0;
    int32_t ytox_dc = 0;
    int32_t ytob_dc = 0;
    uint32_t epf_iters = 0;
    bool gab = false;
    uint32_t num_block_ctxs = 0;
    uint32_t num_qf_thresholds = 0;
    bool dequant_all_default = true;
    bool dc_quant_all_default = true;
    bool all_default = true;
    float dc_quant_weights[3] = {kDCQuant[0], kDCQuant[1], kDCQuant[2]};
    float dct8_dequant_mean[3] = {0.0f, 0.0f, 0.0f};
  };

  std::vector<FrameInspectInfo> frames;
  size_t frame_idx = 0;
  while (pos < compressed.size()) {
    FrameHeader fh(&metadata);
    auto decoded = jxl::make_unique<ImageBundle>(memory_manager, &metadata.m);
    JXL_RETURN_IF_ERROR(DecodeFrame(dec_state.get(), pool.get(),
                                    compressed.data() + pos,
                                    compressed.size() - pos, &fh, decoded.get(),
                                    metadata));
    JXL_ENSURE(decoded->decoded_bytes() > 0);
    pos += decoded->decoded_bytes();

    FrameDimensions fdim = fh.ToFrameDimensions();
    FrameInspectInfo info;
    info.index = frame_idx++;
    info.frame_type = FrameTypeName(fh.frame_type);
    info.frame_type_id = static_cast<uint32_t>(fh.frame_type);
    info.encoding =
        (fh.encoding == FrameEncoding::kVarDCT) ? "kVarDCT" : "kModular";
    info.width = fdim.xsize;
    info.height = fdim.ysize;
    info.upsampling = fh.upsampling;
    for (size_t i = 0; i < fh.extra_channel_upsampling.size(); ++i) {
      uint32_t dshift = metadata.m.extra_channel_info[i].dim_shift;
      info.effective_ec_upsampling.push_back(fh.extra_channel_upsampling[i]);
      info.ec_upsampling.push_back(fh.extra_channel_upsampling[i] >> dshift);
    }
    info.save_before_color_transform = fh.save_before_color_transform;
    info.save_as_reference = fh.save_as_reference;
    info.is_last = fh.is_last;
    info.has_noise = (fh.flags & FrameHeader::kNoise) != 0;
    for (size_t i = 0; i < 8; ++i) {
      info.noise_lut[i] =
          info.has_noise
              ? dec_state->shared->image_features.noise_params.lut[i]
              : 0.0f;
    }
    info.has_patches = (fh.flags & FrameHeader::kPatches) != 0;
    if (info.has_patches) {
      const PatchDictionary& pdic = dec_state->shared->image_features.patches;
      const auto& pos_vec = GetPatchPositions(pdic);
      const auto& ref_vec = GetPatchRefPositions(pdic);
      const auto& blend_vec = GetPatchBlendings(pdic);
      const size_t b_stride = GetPatchBlendingsStride(pdic);
      info.num_patches = pos_vec.size();
      info.num_ref_patches = ref_vec.size();
      if (b_stride > 0) {
        for (size_t p = 0; p < pos_vec.size(); ++p) {
          info.patch_blend_modes.push_back(
              static_cast<uint32_t>(blend_vec[p * b_stride].mode));
          info.patch_alpha_channels.push_back(
              static_cast<uint32_t>(blend_vec[p * b_stride].alpha_channel));
          info.patch_clamps.push_back(blend_vec[p * b_stride].clamp);
          if (b_stride > 1) {
            info.patch_ec_blend_modes.push_back(
                static_cast<uint32_t>(blend_vec[p * b_stride + 1].mode));
          }
        }
      }
    }
    info.blending_mode = BlendModeName(fh.blending_info.mode);
    info.blending_mode_id = static_cast<uint32_t>(fh.blending_info.mode);
    info.blending_source = fh.blending_info.source;
    info.alpha_channel = fh.blending_info.alpha_channel;

    QuantizerParams qp = dec_state->shared->quantizer.GetParams();
    info.global_scale = static_cast<int32_t>(qp.global_scale);
    info.quant_dc = static_cast<int32_t>(qp.quant_dc);
    info.x_qm_scale = fh.x_qm_scale;
    info.b_qm_scale = fh.b_qm_scale;
    info.ytox_dc = dec_state->shared->cmap.base().GetYToXDC();
    info.ytob_dc = dec_state->shared->cmap.base().GetYToBDC();
    info.epf_iters = fh.loop_filter.epf_iters;
    info.gab = fh.loop_filter.gab;
    info.num_block_ctxs =
        static_cast<uint32_t>(dec_state->shared->block_ctx_map.num_ctxs);
    info.num_qf_thresholds = static_cast<uint32_t>(
        dec_state->shared->block_ctx_map.qf_thresholds.size());

    const auto& encodings = dec_state->shared->matrices.encodings();
    info.dequant_all_default = true;
    for (const auto& enc : encodings) {
      if (enc.mode != QuantEncoding::kQuantModeLibrary || enc.predefined != 0) {
        info.dequant_all_default = false;
        break;
      }
    }
    info.dc_quant_all_default = true;
    for (size_t c = 0; c < 3; ++c) {
      info.dc_quant_weights[c] = dec_state->shared->matrices.DCQuants()[c];
      if (std::abs(info.dc_quant_weights[c] - kDCQuant[c]) > 1e-7f) {
        info.dc_quant_all_default = false;
      }
    }
    info.all_default = info.dequant_all_default && info.dc_quant_all_default;
    if (fh.encoding == FrameEncoding::kVarDCT) {
      JXL_RETURN_IF_ERROR(
          dec_state->shared_storage.matrices.EnsureComputed(memory_manager, 1u));
      for (size_t c = 0; c < 3; ++c) {
        const float* mat =
            dec_state->shared_storage.matrices.Matrix(AcStrategyType::DCT, c);
        double sum = 0.0;
        for (size_t k = 1; k < 64; ++k) sum += mat[k];
        info.dct8_dequant_mean[c] = static_cast<float>(sum / 63.0);
      }
    }

    frames.push_back(std::move(info));
    if (fh.is_last) break;
  }

  JXL_ENSURE(!frames.empty());
  const FrameInspectInfo& last = frames.back();
  bool any_frame_non_default_dequant = false;
  bool any_frame_non_default_dc = false;
  for (const auto& fr : frames) {
    if (!fr.dequant_all_default) any_frame_non_default_dequant = true;
    if (!fr.dc_quant_all_default) any_frame_non_default_dc = true;
  }
  const bool top_dequant_all_default = !any_frame_non_default_dequant;
  const bool top_dc_quant_all_default = !any_frame_non_default_dc;
  const bool top_all_default =
      top_dequant_all_default && top_dc_quant_all_default;

  std::ostringstream oss;
  oss << std::setprecision(7);
  oss << "{\n";
  oss << "  \"file_bytes\": " << compressed.size() << ",\n";
  oss << "  \"width\": " << metadata.size.xsize() << ",\n";
  oss << "  \"height\": " << metadata.size.ysize() << ",\n";
  oss << "  \"xsize\": " << metadata.size.xsize() << ",\n";
  oss << "  \"ysize\": " << metadata.size.ysize() << ",\n";
  oss << "  \"xsize_blocks\": " << DivCeil(metadata.size.xsize(), 8) << ",\n";
  oss << "  \"ysize_blocks\": " << DivCeil(metadata.size.ysize(), 8) << ",\n";
  oss << "  \"num_frames\": " << frames.size() << ",\n";
  oss << "  \"encoding\": \""
      << (last.encoding == "kVarDCT" ? "VarDCT" : "Modular") << "\",\n";
  oss << "  \"global_scale\": " << last.global_scale << ",\n";
  oss << "  \"quant_dc\": " << last.quant_dc << ",\n";
  oss << "  \"all_default\": " << (top_all_default ? "true" : "false") << ",\n";
  oss << "  \"dequant_all_default\": "
      << (top_dequant_all_default ? "true" : "false") << ",\n";
  oss << "  \"dc_quant_all_default\": "
      << (top_dc_quant_all_default ? "true" : "false") << ",\n";
  oss << "  \"dc_quant\": [" << last.dc_quant_weights[0] << ", "
      << last.dc_quant_weights[1] << ", " << last.dc_quant_weights[2]
      << "],\n";
  oss << "  \"dc_quant_weights\": [" << last.dc_quant_weights[0] << ", "
      << last.dc_quant_weights[1] << ", " << last.dc_quant_weights[2]
      << "],\n";
  oss << "  \"dct8_dequant_mean\": [" << last.dct8_dequant_mean[0] << ", "
      << last.dct8_dequant_mean[1] << ", " << last.dct8_dequant_mean[2]
      << "],\n";
  oss << "  \"x_qm_scale\": " << last.x_qm_scale << ",\n";
  oss << "  \"b_qm_scale\": " << last.b_qm_scale << ",\n";
  oss << "  \"ytox_dc\": " << last.ytox_dc << ",\n";
  oss << "  \"ytob_dc\": " << last.ytob_dc << ",\n";
  oss << "  \"epf_iters\": " << last.epf_iters << ",\n";
  oss << "  \"gab\": " << (last.gab ? "true" : "false") << ",\n";
  oss << "  \"num_block_ctxs\": " << last.num_block_ctxs << ",\n";
  oss << "  \"num_qf_thresholds\": " << last.num_qf_thresholds << ",\n";
  oss << "  \"has_noise\": " << (last.has_noise ? "true" : "false") << ",\n";
  oss << "  \"noise_lut\": [";
  for (size_t i = 0; i < 8; ++i) {
    oss << last.noise_lut[i] << (i + 1 < 8 ? ", " : "");
  }
  oss << "],\n";
  oss << "  \"has_patches\": " << (last.has_patches ? "true" : "false")
      << ",\n";
  oss << "  \"num_patches\": " << last.num_patches << ",\n";
  oss << "  \"num_extra_channels\": " << metadata.m.extra_channel_info.size()
      << ",\n";
  oss << "  \"alpha_dim_shift\": "
      << (metadata.m.extra_channel_info.empty()
              ? 0
              : metadata.m.extra_channel_info[0].dim_shift)
      << ",\n";
  oss << "  \"extra_channels\": [\n";
  for (size_t i = 0; i < metadata.m.extra_channel_info.size(); ++i) {
    const auto& eci = metadata.m.extra_channel_info[i];
    oss << "    {\n";
    oss << "      \"type\": \"" << ExtraChannelTypeName(eci.type) << "\",\n";
    oss << "      \"bits_per_sample\": " << eci.bit_depth.bits_per_sample
        << ",\n";
    oss << "      \"dim_shift\": " << eci.dim_shift << ",\n";
    oss << "      \"alpha_associated\": "
        << (eci.alpha_associated ? "true" : "false") << "\n";
    oss << "    }"
        << (i + 1 < metadata.m.extra_channel_info.size() ? ",\n" : "\n");
  }
  oss << "  ],\n";
  oss << "  \"frames\": [\n";
  for (size_t f = 0; f < frames.size(); ++f) {
    const auto& fr = frames[f];
    oss << "    {\n";
    oss << "      \"index\": " << fr.index << ",\n";
    oss << "      \"frame_type\": \"" << fr.frame_type << "\",\n";
    oss << "      \"frame_type_id\": " << fr.frame_type_id << ",\n";
    oss << "      \"encoding\": \"" << fr.encoding << "\",\n";
    oss << "      \"width\": " << fr.width << ",\n";
    oss << "      \"height\": " << fr.height << ",\n";
    oss << "      \"upsampling\": " << fr.upsampling << ",\n";
    oss << "      \"ec_upsampling\": [";
    for (size_t i = 0; i < fr.ec_upsampling.size(); ++i) {
      oss << fr.ec_upsampling[i]
          << (i + 1 < fr.ec_upsampling.size() ? ", " : "");
    }
    oss << "],\n";
    oss << "      \"effective_ec_upsampling\": [";
    for (size_t i = 0; i < fr.effective_ec_upsampling.size(); ++i) {
      oss << fr.effective_ec_upsampling[i]
          << (i + 1 < fr.effective_ec_upsampling.size() ? ", " : "");
    }
    oss << "],\n";
    oss << "      \"save_before_color_transform\": "
        << (fr.save_before_color_transform ? "true" : "false") << ",\n";
    oss << "      \"save_as_reference\": " << fr.save_as_reference << ",\n";
    oss << "      \"is_last\": " << (fr.is_last ? "true" : "false") << ",\n";
    oss << "      \"has_noise\": " << (fr.has_noise ? "true" : "false")
        << ",\n";
    oss << "      \"noise_lut\": [";
    for (size_t i = 0; i < 8; ++i) {
      oss << fr.noise_lut[i] << (i + 1 < 8 ? ", " : "");
    }
    oss << "],\n";
    oss << "      \"has_patches\": " << (fr.has_patches ? "true" : "false")
        << ",\n";
    oss << "      \"num_patches\": " << fr.num_patches << ",\n";
    oss << "      \"num_ref_patches\": " << fr.num_ref_patches << ",\n";
    oss << "      \"patch_blend_modes\": [";
    for (size_t i = 0; i < fr.patch_blend_modes.size(); ++i) {
      oss << fr.patch_blend_modes[i]
          << (i + 1 < fr.patch_blend_modes.size() ? ", " : "");
    }
    oss << "],\n";
    oss << "      \"patch_ec_blend_modes\": [";
    for (size_t i = 0; i < fr.patch_ec_blend_modes.size(); ++i) {
      oss << fr.patch_ec_blend_modes[i]
          << (i + 1 < fr.patch_ec_blend_modes.size() ? ", " : "");
    }
    oss << "],\n";
    oss << "      \"patch_alpha_channels\": [";
    for (size_t i = 0; i < fr.patch_alpha_channels.size(); ++i) {
      oss << fr.patch_alpha_channels[i]
          << (i + 1 < fr.patch_alpha_channels.size() ? ", " : "");
    }
    oss << "],\n";
    oss << "      \"patch_clamps\": [";
    for (size_t i = 0; i < fr.patch_clamps.size(); ++i) {
      oss << (fr.patch_clamps[i] ? "true" : "false")
          << (i + 1 < fr.patch_clamps.size() ? ", " : "");
    }
    oss << "],\n";
    oss << "      \"blending_mode\": \"" << fr.blending_mode << "\",\n";
    oss << "      \"blending_mode_id\": " << fr.blending_mode_id << ",\n";
    oss << "      \"blending_source\": " << fr.blending_source << ",\n";
    oss << "      \"alpha_channel\": " << fr.alpha_channel << ",\n";
    oss << "      \"global_scale\": " << fr.global_scale << ",\n";
    oss << "      \"quant_dc\": " << fr.quant_dc << ",\n";
    oss << "      \"all_default\": " << (fr.all_default ? "true" : "false")
        << ",\n";
    oss << "      \"dequant_all_default\": "
        << (fr.dequant_all_default ? "true" : "false") << ",\n";
    oss << "      \"dc_quant_all_default\": "
        << (fr.dc_quant_all_default ? "true" : "false") << ",\n";
    oss << "      \"dc_quant\": [" << fr.dc_quant_weights[0] << ", "
        << fr.dc_quant_weights[1] << ", " << fr.dc_quant_weights[2] << "],\n";
    oss << "      \"dc_quant_weights\": [" << fr.dc_quant_weights[0] << ", "
        << fr.dc_quant_weights[1] << ", " << fr.dc_quant_weights[2] << "],\n";
    oss << "      \"dct8_dequant_mean\": [" << fr.dct8_dequant_mean[0] << ", "
        << fr.dct8_dequant_mean[1] << ", " << fr.dct8_dequant_mean[2] << "],\n";
    oss << "      \"x_qm_scale\": " << fr.x_qm_scale << ",\n";
    oss << "      \"b_qm_scale\": " << fr.b_qm_scale << ",\n";
    oss << "      \"ytox_dc\": " << fr.ytox_dc << ",\n";
    oss << "      \"ytob_dc\": " << fr.ytob_dc << ",\n";
    oss << "      \"epf_iters\": " << fr.epf_iters << ",\n";
    oss << "      \"gab\": " << (fr.gab ? "true" : "false") << ",\n";
    oss << "      \"num_block_ctxs\": " << fr.num_block_ctxs << ",\n";
    oss << "      \"num_qf_thresholds\": " << fr.num_qf_thresholds << "\n";
    oss << "    }" << (f + 1 < frames.size() ? ",\n" : "\n");
  }
  oss << "  ]\n";
  oss << "}\n";

  std::string json_str = oss.str();
  if (!json_out_path.empty()) {
    if (!WriteTextFile(json_out_path, json_str)) {
      return JXL_FAILURE("Failed to write inspect JSON: %s",
                         json_out_path.c_str());
    }
  }
  std::cout << json_str;
  return true;
}

}  // namespace
}  // namespace jxl

int main(int argc, char** argv) {
  std::string mode;
  std::string input_path;
  std::string output_state_path;
  std::string output_jxl_path;
  std::string json_out_path;
  std::string state_path;
  std::string ref_state_path;
  std::string main_state_path;
  std::string patches_path;
  std::string alpha_path;
  std::string noise_lut_str;
  float distance = 1.0f;
  int effort = 7;
  uint32_t upsampling = 2;
  uint32_t dim_shift = 0;
  uint32_t alpha_blend_mode =
      static_cast<uint32_t>(jxl::PatchBlendMode::kBlendBelow);
  uint32_t alpha_bits = 8;
  bool alpha_clamp = false;

  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--dump-init" || arg == "--encode-single" ||
        arg == "--encode-patches" || arg == "--encode-alpha" ||
        arg == "--inspect") {
      mode = arg;
    } else if (arg == "--input" && i + 1 < argc) {
      input_path = argv[++i];
    } else if (arg == "--output-state" && i + 1 < argc) {
      output_state_path = argv[++i];
    } else if ((arg == "--output-jxl" || arg == "--output") && i + 1 < argc) {
      output_jxl_path = argv[++i];
    } else if (arg == "--json-out" && i + 1 < argc) {
      json_out_path = argv[++i];
    } else if (arg == "--state" && i + 1 < argc) {
      state_path = argv[++i];
    } else if (arg == "--ref-state" && i + 1 < argc) {
      ref_state_path = argv[++i];
    } else if (arg == "--main-state" && i + 1 < argc) {
      main_state_path = argv[++i];
    } else if (arg == "--patches" && i + 1 < argc) {
      patches_path = argv[++i];
    } else if (arg == "--alpha" && i + 1 < argc) {
      alpha_path = argv[++i];
    } else if (arg == "--noise-lut" && i + 1 < argc) {
      noise_lut_str = argv[++i];
    } else if (arg == "--distance" && i + 1 < argc) {
      distance = std::strtof(argv[++i], nullptr);
    } else if (arg == "--effort" && i + 1 < argc) {
      effort = std::atoi(argv[++i]);
    } else if (arg == "--upsampling" && i + 1 < argc) {
      upsampling = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if ((arg == "--dim-shift" || arg == "--alpha-dim-shift") &&
               i + 1 < argc) {
      dim_shift = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (arg == "--alpha-blend-mode" && i + 1 < argc) {
      alpha_blend_mode = jxl::ParseAlphaBlendModeArg(argv[++i]);
    } else if (arg == "--alpha-bits" && i + 1 < argc) {
      alpha_bits = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (arg == "--alpha-clamp" && i + 1 < argc) {
      alpha_clamp = (std::atoi(argv[++i]) != 0);
    } else if (!arg.empty() && arg[0] != '-') {
      positional.push_back(arg);
    } else {
      fprintf(stderr, "Unknown or incomplete argument: %s\n", arg.c_str());
      return 1;
    }
  }

  jxl::Status status = true;
  if (mode == "--dump-init") {
    if (input_path.empty() && positional.size() >= 1)
      input_path = positional[0];
    if (output_state_path.empty() && positional.size() >= 2)
      output_state_path = positional[1];
    if (json_out_path.empty() && positional.size() >= 3)
      json_out_path = positional[2];
    if (input_path.empty() || output_state_path.empty()) {
      fprintf(stderr,
              "Usage: %s --dump-init --input <in.png> --output-state "
              "<state.bin> [--distance <d>] [--effort <e>] [--output-jxl "
              "<init.jxl>] [--json-out <stats.json>]\n",
              argv[0]);
      return 1;
    }
    status = jxl::RunDumpInit(input_path, distance, effort, output_state_path,
                              output_jxl_path, json_out_path);
  } else if (mode == "--encode-single") {
    if (state_path.empty() && positional.size() >= 1)
      state_path = positional[0];
    if (output_jxl_path.empty() && positional.size() >= 2)
      output_jxl_path = positional[1];
    if (json_out_path.empty() && positional.size() >= 3)
      json_out_path = positional[2];
    if (state_path.empty() || output_jxl_path.empty()) {
      fprintf(stderr,
              "Usage: %s --encode-single --state <state.bin> --output-jxl "
              "<out.jxl> [--effort <e>] [--noise-lut <f0,..,f7>] [--json-out "
              "<stats.json>]\n",
              argv[0]);
      return 1;
    }
    status = jxl::RunEncodeSingle(state_path, output_jxl_path, effort,
                                  noise_lut_str, json_out_path);
  } else if (mode == "--encode-patches") {
    if (ref_state_path.empty() || main_state_path.empty() ||
        output_jxl_path.empty()) {
      fprintf(stderr,
              "Usage: %s --encode-patches --ref-state <ref.bin> --main-state "
              "<main.bin> --upsampling <2|4|8> [--patches <patches.bin>] "
              "--output-jxl <out.jxl> [--effort <e>] [--noise-lut <f0,..,f7>] "
              "[--json-out <stats.json>]\n",
              argv[0]);
      return 1;
    }
    status = jxl::RunEncodePatches(ref_state_path, main_state_path, upsampling,
                                   patches_path, output_jxl_path, effort,
                                   noise_lut_str, json_out_path);
  } else if (mode == "--encode-alpha") {
    if (ref_state_path.empty() || main_state_path.empty() ||
        alpha_path.empty() || output_jxl_path.empty()) {
      fprintf(stderr,
              "Usage: %s --encode-alpha --ref-state <ref.bin> --main-state "
              "<main.bin> --upsampling <2|4|8> --dim-shift <0|1|2|3> --alpha "
              "<alpha.bin> [--patches <patches.bin>] [--alpha-blend-mode "
              "<4|5|6|7>] [--alpha-bits <8>] --output-jxl <out.jxl> [--effort "
              "<e>] [--noise-lut <f0,..,f7>] [--json-out <stats.json>]\n",
              argv[0]);
      return 1;
    }
    status = jxl::RunEncodeAlpha(
        ref_state_path, main_state_path, upsampling, dim_shift, alpha_path,
        patches_path, alpha_blend_mode, alpha_bits, alpha_clamp,
        output_jxl_path, effort, noise_lut_str, json_out_path);
  } else if (mode == "--inspect") {
    if (input_path.empty() && positional.size() >= 1)
      input_path = positional[0];
    if (json_out_path.empty() && positional.size() >= 2)
      json_out_path = positional[1];
    if (input_path.empty()) {
      fprintf(
          stderr,
          "Usage: %s --inspect --input <in.jxl> [--json-out <inspect.json>]\n",
          argv[0]);
      return 1;
    }
    status = jxl::RunInspect(input_path, json_out_path);
  } else {
    fprintf(stderr,
            "Specify one of: --dump-init, --encode-single, --encode-patches, "
            "--encode-alpha, --inspect\n");
    return 1;
  }

  if (!status) {
    fprintf(stderr, "jxl_wasserstein_helper failed in mode %s\n", mode.c_str());
    return 1;
  }
  return 0;
}
