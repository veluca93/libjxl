// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Library to encode the context map.

#include "lib/jxl/enc_context_map.h"

#include <jxl/memory_manager.h>
#include <jxl/types.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "lib/jxl/ac_context.h"
#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_ans_params.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/entropy_coder.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/pack_signed.h"

namespace jxl {

namespace {

size_t IndexOf(const std::vector<uint8_t>& v, uint8_t value) {
  size_t i = 0;
  for (; i < v.size(); ++i) {
    if (v[i] == value) return i;
  }
  return i;
}

void MoveToFront(std::vector<uint8_t>* v, size_t index) {
  uint8_t value = (*v)[index];
  for (size_t i = index; i != 0; --i) {
    (*v)[i] = (*v)[i - 1];
  }
  (*v)[0] = value;
}

std::vector<uint8_t> MoveToFrontTransform(const std::vector<uint8_t>& v) {
  if (v.empty()) return v;
  uint8_t max_value = *std::max_element(v.begin(), v.end());
  std::vector<uint8_t> mtf(max_value + 1);
  for (size_t i = 0; i <= max_value; ++i) mtf[i] = i;
  std::vector<uint8_t> result(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    size_t index = IndexOf(mtf, v[i]);
    JXL_DASSERT(index < mtf.size());
    result[i] = static_cast<uint8_t>(index);
    MoveToFront(&mtf, index);
  }
  return result;
}

}  // namespace

Status EncodeContextMap(const std::vector<uint8_t>& context_map,
                        size_t num_histograms, BitWriter* writer,
                        LayerType layer, AuxOut* aux_out,
                        bool use_rdo_options) {
  if (num_histograms == 1) {
    // Simple code
    writer->Write(1, 1);
    // 0 bits per entry.
    writer->Write(2, 0);
    return true;
  }

  JxlMemoryManager* memory_manager = writer->memory_manager();
  std::vector<uint8_t> transformed_symbols = MoveToFrontTransform(context_map);

  if (!use_rdo_options) {
    std::vector<std::vector<Token>> tokens(1);
    std::vector<std::vector<Token>> mtf_tokens(1);
    for (const uint8_t& ctx : context_map) {
      tokens[0].emplace_back(0, ctx);
    }
    for (const uint8_t& sym : transformed_symbols) {
      mtf_tokens[0].emplace_back(0, sym);
    }
    HistogramParams params;
    params.uint_method = HistogramParams::HybridUintMethod::kContextMap;
    size_t ans_cost;
    size_t mtf_cost;
    {
      EntropyEncodingData codes;
      JXL_ASSIGN_OR_RETURN(
          ans_cost,
          BuildAndEncodeHistograms(memory_manager, params, 1, tokens, &codes,
                                   nullptr, LayerType::Header, nullptr));
    }
    {
      EntropyEncodingData codes;
      JXL_ASSIGN_OR_RETURN(
          mtf_cost, BuildAndEncodeHistograms(memory_manager, params, 1,
                                             mtf_tokens, &codes, nullptr,
                                             LayerType::Header, nullptr));
    }
    const bool use_mtf = mtf_cost < ans_cost;
    tokens[0].clear();
    for (size_t i = 0; i < transformed_symbols.size(); i++) {
      tokens[0].emplace_back(0,
                             use_mtf ? transformed_symbols[i] : context_map[i]);
    }
    const size_t entry_bits = CeilLog2Nonzero(num_histograms);
    const size_t simple_cost = entry_bits * context_map.size();
    if (entry_bits < 4 && simple_cost < ans_cost && simple_cost < mtf_cost) {
      return writer->WithMaxBits(
          3 + entry_bits * context_map.size(), layer, aux_out, [&] {
            writer->Write(1, 1);
            writer->Write(2, entry_bits);
            for (uint8_t entry : context_map) {
              writer->Write(entry_bits, entry);
            }
            return true;
          });
    }
    return writer->WithMaxBits(
        2 + tokens[0].size() * 24, layer, aux_out, [&]() -> Status {
          writer->Write(1, 0);
          writer->Write(1, TO_JXL_BOOL(use_mtf));
          EntropyEncodingData codes;
          JXL_ASSIGN_OR_RETURN(
              size_t cost,
              BuildAndEncodeHistograms(memory_manager, params, 1, tokens,
                                       &codes, writer, layer, aux_out));
          (void)cost;
          WriteTokens(tokens[0], codes, 0, writer);
          return true;
        });
  }

  size_t entry_bits = CeilLog2Nonzero(num_histograms);
  size_t simple_cost = 3 + entry_bits * context_map.size();

  struct CandOption {
    bool use_mtf;
    HistogramParams::LZ77Method lz77;
    HistogramParams::HybridUintMethod uint_method;
    HistogramParams::ANSHistogramStrategy ans_strategy;
    HistogramParams::ClusteringType clustering;
  };

  std::vector<CandOption> options;
  const auto base_lz77 = (context_map.size() <= 2)
                             ? HistogramParams::LZ77Method::kNone
                             : HistogramParams::LZ77Method::kRLE;
  for (bool mtf : {false, true}) {
    options.push_back(
        {mtf, base_lz77, HistogramParams::HybridUintMethod::kContextMap,
         HistogramParams::ANSHistogramStrategy::kPrecise,
         HistogramParams::ClusteringType::kBest});
  }
  if (context_map.size() > 64) {
    for (bool mtf : {false, true}) {
      options.push_back(
          {mtf, HistogramParams::LZ77Method::kRLE,
           HistogramParams::HybridUintMethod::kBest,
           HistogramParams::ANSHistogramStrategy::kPrecise,
           HistogramParams::ClusteringType::kBest});
      options.push_back(
          {mtf, HistogramParams::LZ77Method::kOptc256,
           HistogramParams::HybridUintMethod::kBest,
           HistogramParams::ANSHistogramStrategy::kPrecise,
           HistogramParams::ClusteringType::kBest});
    }
  }

  size_t best_bits = std::numeric_limits<size_t>::max();
  size_t best_opt = 0;
  for (size_t idx = 0; idx < options.size(); ++idx) {
    const CandOption& opt = options[idx];
    const std::vector<uint8_t>& src =
        opt.use_mtf ? transformed_symbols : context_map;
    std::vector<std::vector<Token>> cand_tokens(1);
    cand_tokens[0].reserve(src.size());
    for (uint8_t sym : src) {
      cand_tokens[0].emplace_back(0, sym);
    }
    HistogramParams cand_params;
    cand_params.lz77_method = opt.lz77;
    cand_params.uint_method = opt.uint_method;
    cand_params.ans_histogram_strategy = opt.ans_strategy;
    cand_params.clustering = opt.clustering;
    BitWriter cand_writer{memory_manager};
    JXL_RETURN_IF_ERROR(cand_writer.WithMaxBits(
        2 + src.size() * 24 + 1024, LayerType::Header, nullptr,
        [&]() -> Status {
          cand_writer.Write(1, 0);
          cand_writer.Write(1, TO_JXL_BOOL(opt.use_mtf));
          EntropyEncodingData codes;
          JXL_ASSIGN_OR_RETURN(
              size_t cost, BuildAndEncodeHistograms(memory_manager, cand_params,
                                                    1, cand_tokens, &codes,
                                                    &cand_writer,
                                                    LayerType::Header, nullptr));
          (void)cost;
          WriteTokens(cand_tokens[0], codes, 0, &cand_writer);
          return true;
        }));
    if (cand_writer.BitsWritten() < best_bits) {
      best_bits = cand_writer.BitsWritten();
      best_opt = idx;
    }
  }

  if (entry_bits < 4 && simple_cost < best_bits) {
    JXL_RETURN_IF_ERROR(writer->WithMaxBits(
        simple_cost, layer, aux_out, [&] {
          writer->Write(1, 1);
          writer->Write(2, entry_bits);
          for (uint8_t entry : context_map) {
            writer->Write(entry_bits, entry);
          }
          return true;
        }));
  } else {
    const CandOption& opt = options[best_opt];
    const std::vector<uint8_t>& src =
        opt.use_mtf ? transformed_symbols : context_map;
    std::vector<std::vector<Token>> tokens(1);
    tokens[0].reserve(src.size());
    for (uint8_t sym : src) {
      tokens[0].emplace_back(0, sym);
    }
    HistogramParams params;
    params.lz77_method = opt.lz77;
    params.uint_method = opt.uint_method;
    params.ans_histogram_strategy = opt.ans_strategy;
    params.clustering = opt.clustering;
    JXL_RETURN_IF_ERROR(writer->WithMaxBits(
        2 + src.size() * 24 + 1024, layer, aux_out, [&]() -> Status {
          writer->Write(1, 0);
          writer->Write(1, TO_JXL_BOOL(opt.use_mtf));  // Use/don't use MTF.
          EntropyEncodingData codes;
          JXL_ASSIGN_OR_RETURN(
              size_t cost,
              BuildAndEncodeHistograms(memory_manager, params, 1, tokens,
                                       &codes, writer, layer, aux_out));
          (void)cost;
          WriteTokens(tokens[0], codes, 0, writer);
          return true;
        }));
  }
  return true;
}

Status EncodeBlockCtxMap(const BlockCtxMap& block_ctx_map, BitWriter* writer,
                         AuxOut* aux_out) {
  const auto& dct = block_ctx_map.dc_thresholds;
  const auto& qft = block_ctx_map.qf_thresholds;
  const auto& ctx_map = block_ctx_map.ctx_map;
  return writer->WithMaxBits(
      (dct[0].size() + dct[1].size() + dct[2].size() + qft.size()) * 34 + 1 +
          4 + 4 + ctx_map.size() * 10 + 1024,
      LayerType::Ac, aux_out, [&]() -> Status {
        if (dct[0].empty() && dct[1].empty() && dct[2].empty() && qft.empty() &&
            ctx_map.size() == sizeof(BlockCtxMap::kDefaultCtxMap) &&
            std::equal(ctx_map.begin(), ctx_map.end(),
                       BlockCtxMap::kDefaultCtxMap)) {
          writer->Write(1, 1);  // default
          return true;
        }
        writer->Write(1, 0);
        for (int j : {0, 1, 2}) {
          writer->Write(4, dct[j].size());
          for (int i : dct[j]) {
            JXL_RETURN_IF_ERROR(
                U32Coder::Write(kDCThresholdDist, PackSigned(i), writer));
          }
        }
        writer->Write(4, qft.size());
        for (uint32_t i : qft) {
          JXL_RETURN_IF_ERROR(U32Coder::Write(kQFThresholdDist, i - 1, writer));
        }
        JXL_RETURN_IF_ERROR(EncodeContextMap(ctx_map, block_ctx_map.num_ctxs,
                                             writer, LayerType::Ac, aux_out));
        return true;
      });
}

}  // namespace jxl
