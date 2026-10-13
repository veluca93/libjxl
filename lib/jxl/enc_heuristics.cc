// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_heuristics.h"

#include <jxl/cms_interface.h>
#include <jxl/memory_manager.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "lib/jxl/ac_context.h"
#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/override.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/butteraugli/butteraugli.h"
#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/coeff_order.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/common.h"
#include "lib/jxl/dct_util.h"
#include "lib/jxl/dec_cache.h"
#include "lib/jxl/dec_group.h"
#include "lib/jxl/dec_noise.h"
#include "lib/jxl/dec_xyb.h"
#include "lib/jxl/enc_ac_strategy.h"
#include "lib/jxl/enc_adaptive_quantization.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_chroma_from_luma.h"
#include "lib/jxl/enc_gaborish.h"
#include "lib/jxl/enc_modular.h"
#include "lib/jxl/enc_noise.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/enc_patch_dictionary.h"
#include "lib/jxl/enc_quant_weights.h"
#include "lib/jxl/enc_splines.h"
#include "lib/jxl/epf.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/passes_state.h"
#include "lib/jxl/quant_weights.h"
#include "lib/jxl/render_pipeline/render_pipeline.h"

namespace jxl {

struct AuxOut;

void FindBestBlockEntropyModel(const CompressParams& cparams, const ImageI& rqf,
                               const AcStrategyImage& ac_strategy,
                               BlockCtxMap* block_ctx_map) {
  if (cparams.decoding_speed_tier >= 1) {
    static constexpr uint8_t kSimpleCtxMap[] = {
        // Cluster all blocks together
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  //
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  //
    };
    static_assert(
        3 * kNumOrders == sizeof(kSimpleCtxMap) / sizeof *kSimpleCtxMap,
        "Update simple context map");

    auto& bcm = *block_ctx_map;
    bcm.ctx_map.assign(std::begin(kSimpleCtxMap), std::end(kSimpleCtxMap));
    bcm.num_ctxs = 2;
    bcm.num_dc_ctxs = 1;
    return;
  }
  if (cparams.speed_tier >= SpeedTier::kFalcon) {
    return;
  }
  // No need to change context modeling for small images.
  size_t tot = rqf.xsize() * rqf.ysize();
  size_t size_for_ctx_model = (1 << 10) * cparams.butteraugli_distance;
  if (tot < size_for_ctx_model) return;

  struct OccCounters {
    // count the occurrences of each qf value and each strategy type.
    OccCounters(const ImageI& rqf, const AcStrategyImage& ac_strategy) {
      for (size_t y = 0; y < rqf.ysize(); y++) {
        const int32_t* qf_row = rqf.Row(y);
        AcStrategyRow acs_row = ac_strategy.ConstRow(y);
        for (size_t x = 0; x < rqf.xsize(); x++) {
          int ord = kStrategyOrder[acs_row[x].RawStrategy()];
          int qf = qf_row[x] - 1;
          qf_counts[qf]++;
          qf_ord_counts[ord][qf]++;
          ord_counts[ord]++;
        }
      }
    }

    size_t qf_counts[256] = {};
    size_t qf_ord_counts[kNumOrders][256] = {};
    size_t ord_counts[kNumOrders] = {};
  };
  // The OccCounters struct is too big to allocate on the stack.
  std::unique_ptr<OccCounters> counters(new OccCounters(rqf, ac_strategy));

  // Splitting the context model according to the quantization field seems to
  // mostly benefit only large images.
  size_t size_for_qf_split = (1 << 13) * cparams.butteraugli_distance;
  size_t num_qf_segments = tot < size_for_qf_split ? 1 : 2;
  std::vector<uint32_t>& qft = block_ctx_map->qf_thresholds;
  qft.clear();
  // Divide the quant field in up to num_qf_segments segments.
  size_t cumsum = 0;
  size_t next = 1;
  size_t last_cut = 256;
  size_t cut = tot * next / num_qf_segments;
  for (uint32_t j = 0; j < 256; j++) {
    cumsum += counters->qf_counts[j];
    if (cumsum > cut) {
      if (j != 0) {
        qft.push_back(j);
      }
      last_cut = j;
      while (cumsum > cut) {
        next++;
        cut = tot * next / num_qf_segments;
      }
    } else if (next > qft.size() + 1) {
      if (j - 1 == last_cut && j != 0) {
        qft.push_back(j);
      }
    }
  }

  // Count the occurrences of each segment.
  std::vector<size_t> counts(kNumOrders * (qft.size() + 1));
  size_t qft_pos = 0;
  for (size_t j = 0; j < 256; j++) {
    if (qft_pos < qft.size() && j == qft[qft_pos]) {
      qft_pos++;
    }
    for (size_t i = 0; i < kNumOrders; i++) {
      counts[qft_pos + i * (qft.size() + 1)] += counters->qf_ord_counts[i][j];
    }
  }

  // Repeatedly merge the lowest-count pair.
  std::vector<uint8_t> remap((qft.size() + 1) * kNumOrders);
  std::iota(remap.begin(), remap.end(), 0);
  std::vector<uint8_t> clusters(remap);
  size_t nb_clusters =
      Clamp1(static_cast<int>(tot / size_for_ctx_model / 2), 2, 9);
  size_t nb_clusters_chroma =
      Clamp1(static_cast<int>(tot / size_for_ctx_model / 3), 1, 5);
  // This is O(n^2 log n), but n is small.
  while (clusters.size() > nb_clusters) {
    std::sort(clusters.begin(), clusters.end(),
              [&](int a, int b) { return counts[a] > counts[b]; });
    counts[clusters[clusters.size() - 2]] += counts[clusters.back()];
    counts[clusters.back()] = 0;
    remap[clusters.back()] = clusters[clusters.size() - 2];
    clusters.pop_back();
  }
  for (size_t i = 0; i < remap.size(); i++) {
    while (remap[remap[i]] != remap[i]) {
      remap[i] = remap[remap[i]];
    }
  }
  // Sort remaining cluster roots by descending count so the largest clusters
  // receive the lowest indices (and thus remain unmerged for chroma below).
  std::sort(clusters.begin(), clusters.end(),
            [&](int a, int b) { return counts[a] > counts[b]; });
  // Relabel starting from 0 in descending cluster size order.
  std::vector<uint8_t> remap_remap(remap.size(), remap.size());
  size_t num = 0;
  for (uint8_t c : clusters) {
    remap_remap[c] = num++;
  }
  for (size_t i = 0; i < remap.size(); i++) {
    if (remap_remap[remap[i]] == remap.size()) {
      remap_remap[remap[i]] = num++;
    }
    remap[i] = remap_remap[remap[i]];
  }
  // Write the block context map.
  auto& ctx_map = block_ctx_map->ctx_map;
  ctx_map = remap;
  ctx_map.resize(remap.size() * 3);
  // for chroma, only use up to nb_clusters_chroma separate block contexts
  // (those for the biggest clusters)
  for (size_t i = remap.size(); i < remap.size() * 3; i++) {
    ctx_map[i] = num + Clamp1(static_cast<int>(remap[i % remap.size()]), 0,
                              static_cast<int>(nb_clusters_chroma) - 1);
  }
  block_ctx_map->num_ctxs =
      *std::max_element(ctx_map.begin(), ctx_map.end()) + 1;
}

namespace {

Status FindBestDequantMatrices(JxlMemoryManager* memory_manager,
                               const CompressParams& cparams,
                               const FrameDimensions& frame_dim,
                               bool enable_rdo,
                               ModularFrameEncoder* modular_frame_encoder,
                               DequantMatrices* dequant_matrices) {
  // TODO(veluca): quant matrices for no-gaborish.
  // TODO(veluca): heuristics for in-bitstream quant tables.
  *dequant_matrices = DequantMatrices();
  if (cparams.max_error_mode || cparams.disable_perceptual_optimizations) {
    constexpr float kMSEWeights[3] = {0.001, 0.001, 0.001};
    const float* wp = cparams.disable_perceptual_optimizations
                          ? kMSEWeights
                          : cparams.max_error;
    // Set numerators of all quantization matrices to constant values.
    float weights[3][1] = {{1.0f / wp[0]}, {1.0f / wp[1]}, {1.0f / wp[2]}};
    DctQuantWeightParams dct_params(weights);
    std::vector<QuantEncoding> encodings(kNumQuantTables,
                                         QuantEncoding::DCT(dct_params));
    JXL_RETURN_IF_ERROR(DequantMatricesSetCustom(dequant_matrices, encodings,
                                                 modular_frame_encoder));
    float dc_weights[3] = {1.0f / wp[0], 1.0f / wp[1], 1.0f / wp[2]};
    JXL_RETURN_IF_ERROR(DequantMatricesSetCustomDC(
        memory_manager, dequant_matrices, dc_weights));
  } else if (enable_rdo) {
    const size_t num_blocks = frame_dim.xsize_blocks * frame_dim.ysize_blocks;
    if (num_blocks < 8192) return true;

    const float d = Clamp1(cparams.butteraugli_distance, 0.35f, 3.5f);
    const float t_d = Clamp1((d - 0.5f) / 2.0f, 0.0f, 1.0f);
    const float a_mid_base =
        ((num_blocks >= 16384 && num_blocks < 40000) ? 0.034f : 0.032f) -
        0.008f * t_d;
    constexpr float kChannelMid[3] = {0.75f, 1.00f, 0.85f};

    // Scale both the set of customized transform tables and the number of
    // radial distance bands written to the bitstream with image size
    // (num_blocks) and transform size (lib_bands):
    //   < 8192 blocks   : 0 custom bands (library defaults)
    //   8192..16383     : DCT(6) + DCT16X16(7) + DCT8X16(7) = 20 bands
    //   16384..27999    : DCT(6) + DCT16X16(7) + DCT32X32(8) = 21 bands
    //   28000..39999    : + DCT8X16(7) = 28 bands
    //   >= 40000 blocks : + DCT16X32(8) = 36 bands
    auto target_bands_for_table = [&](QuantTable table,
                                      size_t lib_bands) -> size_t {
      switch (table) {
        case QuantTable::DCT:
        case QuantTable::DCT16X16:
          return lib_bands;
        case QuantTable::DCT8X16:
          if (num_blocks >= 16384 && num_blocks < 28000) return 0;
          return lib_bands;
        case QuantTable::DCT32X32:
          if (num_blocks < 16384) return 0;
          return lib_bands;
        case QuantTable::DCT16X32:
          if (num_blocks < 40000) return 0;
          return lib_bands;
        default:
          return 0;
      }
    };

    auto mult_step = [](float v) -> float {
      return (v > 0.0f) ? (1.0f + v) : (1.0f / (1.0f - v));
    };
    auto inv_mult_step = [](float r) -> float {
      return (r >= 1.0f) ? (r - 1.0f) : (1.0f - 1.0f / r);
    };

    const QuantEncoding* library = DequantMatrices::Library();
    std::vector<QuantEncoding> encodings(kNumQuantTables,
                                         QuantEncoding::Library<0>());
    bool any_custom = false;
    for (size_t i = 0; i < kNumQuantTables; ++i) {
      const QuantTable table = static_cast<QuantTable>(i);
      const QuantEncoding& lib_enc = library[i];
      if (lib_enc.mode != QuantEncoding::kQuantModeDCT) continue;
      const size_t lib_bands = lib_enc.dct_params.num_distance_bands;
      if (lib_bands < 2) continue;
      const size_t out_bands =
          std::min(target_bands_for_table(table, lib_bands), lib_bands);
      if (out_bands < 2) continue;

      float tx_scale = 1.0f;
      if (table == QuantTable::DCT16X16 || table == QuantTable::DCT8X16) {
        tx_scale = 0.85f;
      } else if (table == QuantTable::DCT32X32 ||
                 table == QuantTable::DCT16X32) {
        tx_scale = 0.75f;
      }

      DctQuantWeightParams new_params = lib_enc.dct_params;
      new_params.num_distance_bands = out_bands;
      for (size_t c = 0; c < 3; ++c) {
        float lib_w[DctQuantWeightParams::kMaxDistanceBands];
        lib_w[0] = lib_enc.dct_params.distance_bands[c][0];
        for (size_t j = 1; j < lib_bands; ++j) {
          lib_w[j] =
              lib_w[j - 1] * mult_step(lib_enc.dct_params.distance_bands[c][j]);
        }

        const float a_mid = a_mid_base * kChannelMid[c] * tx_scale;
        float target_w[DctQuantWeightParams::kMaxDistanceBands];
        for (size_t k = 0; k < out_bands; ++k) {
          const float u =
              static_cast<float>(k) / static_cast<float>(out_bands - 1);
          const float pos = u * static_cast<float>(lib_bands - 1);
          const size_t idx =
              std::min<size_t>(static_cast<size_t>(pos), lib_bands - 2);
          const float frac = pos - static_cast<float>(idx);
          const float w_u =
              lib_w[idx] * std::pow(lib_w[idx + 1] / lib_w[idx], frac);
          float delta_l = 0.0f;
          if (k >= 2 && out_bands > 2) {
            const float v =
                static_cast<float>(k - 1) / static_cast<float>(out_bands - 2);
            const float c0 =
                (num_blocks < 16384)
                    ? 0.38f
                    : ((num_blocks >= 40000)
                           ? 0.30f
                           : (1.58f / static_cast<float>(out_bands - 2)));
            const float s = v - c0;
            const float hf_decay =
                (c == 1 || s <= 0.0f) ? 0.45f : ((c == 0) ? 0.85f : 1.05f);
            delta_l = a_mid * s * (1.15f - hf_decay * t_d * v);
          }
          target_w[k] = w_u * std::exp(delta_l);
        }

        new_params.distance_bands[c][0] = target_w[0];
        for (size_t k = 1; k < out_bands; ++k) {
          new_params.distance_bands[c][k] =
              inv_mult_step(target_w[k] / target_w[k - 1]);
        }
        for (size_t k = out_bands; k < DctQuantWeightParams::kMaxDistanceBands;
             ++k) {
          new_params.distance_bands[c][k] = 0.0f;
        }
      }
      encodings[i] = QuantEncoding::DCT(new_params);
      any_custom = true;
    }
    if (any_custom) {
      JXL_RETURN_IF_ERROR(DequantMatricesSetCustom(dequant_matrices, encodings,
                                                   modular_frame_encoder));
    }
  }
  return true;
}

void StoreMin2(const float v, float& min1, float& min2) {
  if (v < min2) {
    if (v < min1) {
      min2 = min1;
      min1 = v;
    } else {
      min2 = v;
    }
  }
}

void CreateMask(const ImageF& image, ImageF& mask) {
  for (size_t y = 0; y < image.ysize(); y++) {
    const auto* row_n = y > 0 ? image.Row(y - 1) : image.Row(y);
    const auto* row_in = image.Row(y);
    const auto* row_s = y + 1 < image.ysize() ? image.Row(y + 1) : image.Row(y);
    auto* row_out = mask.Row(y);
    for (size_t x = 0; x < image.xsize(); x++) {
      // Center, west, east, north, south values and their absolute difference
      float c = row_in[x];
      float w = x > 0 ? row_in[x - 1] : row_in[x];
      float e = x + 1 < image.xsize() ? row_in[x + 1] : row_in[x];
      float n = row_n[x];
      float s = row_s[x];
      float dw = std::abs(c - w);
      float de = std::abs(c - e);
      float dn = std::abs(c - n);
      float ds = std::abs(c - s);
      float min = std::numeric_limits<float>::max();
      float min2 = std::numeric_limits<float>::max();
      StoreMin2(dw, min, min2);
      StoreMin2(de, min, min2);
      StoreMin2(dn, min, min2);
      StoreMin2(ds, min, min2);
      row_out[x] = min2;
    }
  }
}

// Downsamples the image by a factor of 2 with a kernel that's sharper than
// the standard 2x2 box kernel used by DownsampleImage.
// The kernel is optimized against the result of the 2x2 upsampling kernel used
// by the decoder. Ringing is slightly reduced by clamping the values of the
// resulting pixels within certain bounds of a small region in the original
// image.
Status DownsampleImage2_Sharper(const ImageF& input, ImageF* output) {
  const int64_t kernelx = 12;
  const int64_t kernely = 12;
  JxlMemoryManager* memory_manager = input.memory_manager();

  static const float kernel[144] = {
      -0.000314256996835, -0.000314256996835, -0.000897597057705,
      -0.000562751488849, -0.000176807273646, 0.001864627368902,
      0.001864627368902,  -0.000176807273646, -0.000562751488849,
      -0.000897597057705, -0.000314256996835, -0.000314256996835,
      -0.000314256996835, -0.001527942804748, -0.000121760530512,
      0.000191123989093,  0.010193185932466,  0.058637519197110,
      0.058637519197110,  0.010193185932466,  0.000191123989093,
      -0.000121760530512, -0.001527942804748, -0.000314256996835,
      -0.000897597057705, -0.000121760530512, 0.000946363683751,
      0.007113577630288,  0.000437956841058,  -0.000372823835211,
      -0.000372823835211, 0.000437956841058,  0.007113577630288,
      0.000946363683751,  -0.000121760530512, -0.000897597057705,
      -0.000562751488849, 0.000191123989093,  0.007113577630288,
      0.044592622228814,  0.000222278879007,  -0.162864473015945,
      -0.162864473015945, 0.000222278879007,  0.044592622228814,
      0.007113577630288,  0.000191123989093,  -0.000562751488849,
      -0.000176807273646, 0.010193185932466,  0.000437956841058,
      0.000222278879007,  -0.000913092543974, -0.017071696107902,
      -0.017071696107902, -0.000913092543974, 0.000222278879007,
      0.000437956841058,  0.010193185932466,  -0.000176807273646,
      0.001864627368902,  0.058637519197110,  -0.000372823835211,
      -0.162864473015945, -0.017071696107902, 0.414660099370354,
      0.414660099370354,  -0.017071696107902, -0.162864473015945,
      -0.000372823835211, 0.058637519197110,  0.001864627368902,
      0.001864627368902,  0.058637519197110,  -0.000372823835211,
      -0.162864473015945, -0.017071696107902, 0.414660099370354,
      0.414660099370354,  -0.017071696107902, -0.162864473015945,
      -0.000372823835211, 0.058637519197110,  0.001864627368902,
      -0.000176807273646, 0.010193185932466,  0.000437956841058,
      0.000222278879007,  -0.000913092543974, -0.017071696107902,
      -0.017071696107902, -0.000913092543974, 0.000222278879007,
      0.000437956841058,  0.010193185932466,  -0.000176807273646,
      -0.000562751488849, 0.000191123989093,  0.007113577630288,
      0.044592622228814,  0.000222278879007,  -0.162864473015945,
      -0.162864473015945, 0.000222278879007,  0.044592622228814,
      0.007113577630288,  0.000191123989093,  -0.000562751488849,
      -0.000897597057705, -0.000121760530512, 0.000946363683751,
      0.007113577630288,  0.000437956841058,  -0.000372823835211,
      -0.000372823835211, 0.000437956841058,  0.007113577630288,
      0.000946363683751,  -0.000121760530512, -0.000897597057705,
      -0.000314256996835, -0.001527942804748, -0.000121760530512,
      0.000191123989093,  0.010193185932466,  0.058637519197110,
      0.058637519197110,  0.010193185932466,  0.000191123989093,
      -0.000121760530512, -0.001527942804748, -0.000314256996835,
      -0.000314256996835, -0.000314256996835, -0.000897597057705,
      -0.000562751488849, -0.000176807273646, 0.001864627368902,
      0.001864627368902,  -0.000176807273646, -0.000562751488849,
      -0.000897597057705, -0.000314256996835, -0.000314256996835};

  int64_t xsize = input.xsize();
  int64_t ysize = input.ysize();

  JXL_ASSIGN_OR_RETURN(ImageF box_downsample,
                       ImageF::Create(memory_manager, xsize, ysize));
  JXL_RETURN_IF_ERROR(CopyImageTo(input, &box_downsample));
  JXL_ASSIGN_OR_RETURN(box_downsample, DownsampleImage(box_downsample, 2));

  JXL_ASSIGN_OR_RETURN(ImageF mask,
                       ImageF::Create(memory_manager, box_downsample.xsize(),
                                      box_downsample.ysize()));
  CreateMask(box_downsample, mask);

  for (size_t y = 0; y < output->ysize(); y++) {
    float* row_out = output->Row(y);
    const float* row_in[kernely];
    const float* row_mask = mask.Row(y);
    // get the rows in the support
    for (size_t ky = 0; ky < kernely; ky++) {
      int64_t iy = y * 2 + ky - (kernely - 1) / 2;
      if (iy < 0) iy = 0;
      if (iy >= ysize) iy = ysize - 1;
      row_in[ky] = input.Row(iy);
    }

    for (size_t x = 0; x < output->xsize(); x++) {
      // get min and max values of the original image in the support
      float min = std::numeric_limits<float>::max();
      float max = std::numeric_limits<float>::min();
      // kernelx - R and kernely - R are the radius of a rectangular region in
      // which the values of a pixel are bounded to reduce ringing.
      static constexpr int64_t R = 5;
      for (int64_t ky = R; ky + R < kernely; ky++) {
        for (int64_t kx = R; kx + R < kernelx; kx++) {
          int64_t ix = x * 2 + kx - (kernelx - 1) / 2;
          if (ix < 0) ix = 0;
          if (ix >= xsize) ix = xsize - 1;
          min = std::min<float>(min, row_in[ky][ix]);
          max = std::max<float>(max, row_in[ky][ix]);
        }
      }

      float sum = 0;
      for (int64_t ky = 0; ky < kernely; ky++) {
        for (int64_t kx = 0; kx < kernelx; kx++) {
          int64_t ix = x * 2 + kx - (kernelx - 1) / 2;
          if (ix < 0) ix = 0;
          if (ix >= xsize) ix = xsize - 1;
          sum += row_in[ky][ix] * kernel[ky * kernelx + kx];
        }
      }

      row_out[x] = sum;

      // Clamp the pixel within the value  of a small area to prevent ringning.
      // The mask determines how much to clamp, clamp more to reduce more
      // ringing in smooth areas, clamp less in noisy areas to get more
      // sharpness. Higher mask_multiplier gives less clamping, so less
      // ringing reduction.
      const constexpr float mask_multiplier = 1;
      float a = row_mask[x] * mask_multiplier;
      float clip_min = min - a;
      float clip_max = max + a;
      if (row_out[x] < clip_min) {
        row_out[x] = clip_min;
      } else if (row_out[x] > clip_max) {
        row_out[x] = clip_max;
      }
    }
  }
  return true;
}

}  // namespace

Status DownsampleImage2_Sharper(Image3F* opsin) {
  // Allocate extra space to avoid a reallocation when padding.
  JxlMemoryManager* memory_manager = opsin->memory_manager();
  JXL_ASSIGN_OR_RETURN(
      Image3F downsampled,
      Image3F::Create(memory_manager, DivCeil(opsin->xsize(), 2) + kBlockDim,
                      DivCeil(opsin->ysize(), 2) + kBlockDim));
  JXL_RETURN_IF_ERROR(downsampled.ShrinkTo(downsampled.xsize() - kBlockDim,
                                           downsampled.ysize() - kBlockDim));

  for (size_t c = 0; c < 3; c++) {
    JXL_RETURN_IF_ERROR(
        DownsampleImage2_Sharper(opsin->Plane(c), &downsampled.Plane(c)));
  }
  *opsin = std::move(downsampled);
  return true;
}

namespace {

// The default upsampling kernels used by Upsampler in the decoder.
const constexpr int64_t kSize = 5;

const float kernel00[25] = {
    -0.01716200f, -0.03452303f, -0.04022174f, -0.02921014f, -0.00624645f,
    -0.03452303f, 0.14111091f,  0.28896755f,  0.00278718f,  -0.01610267f,
    -0.04022174f, 0.28896755f,  0.56661550f,  0.03777607f,  -0.01986694f,
    -0.02921014f, 0.00278718f,  0.03777607f,  -0.03144731f, -0.01185068f,
    -0.00624645f, -0.01610267f, -0.01986694f, -0.01185068f, -0.00213539f,
};
const float kernel01[25] = {
    -0.00624645f, -0.01610267f, -0.01986694f, -0.01185068f, -0.00213539f,
    -0.02921014f, 0.00278718f,  0.03777607f,  -0.03144731f, -0.01185068f,
    -0.04022174f, 0.28896755f,  0.56661550f,  0.03777607f,  -0.01986694f,
    -0.03452303f, 0.14111091f,  0.28896755f,  0.00278718f,  -0.01610267f,
    -0.01716200f, -0.03452303f, -0.04022174f, -0.02921014f, -0.00624645f,
};
const float kernel10[25] = {
    -0.00624645f, -0.02921014f, -0.04022174f, -0.03452303f, -0.01716200f,
    -0.01610267f, 0.00278718f,  0.28896755f,  0.14111091f,  -0.03452303f,
    -0.01986694f, 0.03777607f,  0.56661550f,  0.28896755f,  -0.04022174f,
    -0.01185068f, -0.03144731f, 0.03777607f,  0.00278718f,  -0.02921014f,
    -0.00213539f, -0.01185068f, -0.01986694f, -0.01610267f, -0.00624645f,
};
const float kernel11[25] = {
    -0.00213539f, -0.01185068f, -0.01986694f, -0.01610267f, -0.00624645f,
    -0.01185068f, -0.03144731f, 0.03777607f,  0.00278718f,  -0.02921014f,
    -0.01986694f, 0.03777607f,  0.56661550f,  0.28896755f,  -0.04022174f,
    -0.01610267f, 0.00278718f,  0.28896755f,  0.14111091f,  -0.03452303f,
    -0.00624645f, -0.02921014f, -0.04022174f, -0.03452303f, -0.01716200f,
};

// Does exactly the same as the Upsampler in dec_upsampler for 2x2 pixels, with
// default CustomTransformData.
// TODO(lode): use Upsampler instead. However, it requires pre-initialization
// and padding on the left side of the image which requires refactoring the
// other code using this.
void UpsampleImage(const ImageF& input, ImageF* output) {
  int64_t xsize = input.xsize();
  int64_t ysize = input.ysize();
  int64_t xsize2 = output->xsize();
  int64_t ysize2 = output->ysize();
  for (int64_t y = 0; y < ysize2; y++) {
    for (int64_t x = 0; x < xsize2; x++) {
      const auto* kernel = kernel00;
      if ((x & 1) && (y & 1)) {
        kernel = kernel11;
      } else if (x & 1) {
        kernel = kernel10;
      } else if (y & 1) {
        kernel = kernel01;
      }
      float sum = 0;
      int64_t x2 = x / 2;
      int64_t y2 = y / 2;

      // get min and max values of the original image in the support
      float min = std::numeric_limits<float>::max();
      float max = std::numeric_limits<float>::min();

      for (int64_t ky = 0; ky < kSize; ky++) {
        for (int64_t kx = 0; kx < kSize; kx++) {
          int64_t xi = x2 - kSize / 2 + kx;
          int64_t yi = y2 - kSize / 2 + ky;
          if (xi < 0) xi = 0;
          if (xi >= xsize) xi = input.xsize() - 1;
          if (yi < 0) yi = 0;
          if (yi >= ysize) yi = input.ysize() - 1;
          min = std::min<float>(min, input.Row(yi)[xi]);
          max = std::max<float>(max, input.Row(yi)[xi]);
        }
      }

      for (int64_t ky = 0; ky < kSize; ky++) {
        for (int64_t kx = 0; kx < kSize; kx++) {
          int64_t xi = x2 - kSize / 2 + kx;
          int64_t yi = y2 - kSize / 2 + ky;
          if (xi < 0) xi = 0;
          if (xi >= xsize) xi = input.xsize() - 1;
          if (yi < 0) yi = 0;
          if (yi >= ysize) yi = input.ysize() - 1;
          sum += input.Row(yi)[xi] * kernel[ky * kSize + kx];
        }
      }
      output->Row(y)[x] = sum;
      if (output->Row(y)[x] < min) output->Row(y)[x] = min;
      if (output->Row(y)[x] > max) output->Row(y)[x] = max;
    }
  }
}

// Returns the derivative of Upsampler, with respect to input pixel x2, y2, to
// output pixel x, y (ignoring the clamping).
float UpsamplerDeriv(int64_t x2, int64_t y2, int64_t x, int64_t y) {
  const auto* kernel = kernel00;
  if ((x & 1) && (y & 1)) {
    kernel = kernel11;
  } else if (x & 1) {
    kernel = kernel10;
  } else if (y & 1) {
    kernel = kernel01;
  }

  int64_t ix = x / 2;
  int64_t iy = y / 2;
  int64_t kx = x2 - ix + kSize / 2;
  int64_t ky = y2 - iy + kSize / 2;

  // This should not happen.
  if (kx < 0 || kx >= kSize || ky < 0 || ky >= kSize) return 0;

  return kernel[ky * kSize + kx];
}

// Apply the derivative of the Upsampler to the input, reversing the effect of
// its coefficients. The output image is 2x2 times smaller than the input.
void AntiUpsample(const ImageF& input, ImageF* d) {
  int64_t xsize = input.xsize();
  int64_t ysize = input.ysize();
  int64_t xsize2 = d->xsize();
  int64_t ysize2 = d->ysize();
  int64_t k0 = kSize - 1;
  int64_t k1 = kSize;
  for (int64_t y2 = 0; y2 < ysize2; ++y2) {
    auto* row = d->Row(y2);
    for (int64_t x2 = 0; x2 < xsize2; ++x2) {
      int64_t x0 = x2 * 2 - k0;
      if (x0 < 0) x0 = 0;
      int64_t x1 = x2 * 2 + k1 + 1;
      if (x1 > xsize) x1 = xsize;
      int64_t y0 = y2 * 2 - k0;
      if (y0 < 0) y0 = 0;
      int64_t y1 = y2 * 2 + k1 + 1;
      if (y1 > ysize) y1 = ysize;

      float sum = 0;
      for (int64_t y = y0; y < y1; ++y) {
        const auto* row_in = input.Row(y);
        for (int64_t x = x0; x < x1; ++x) {
          double deriv = UpsamplerDeriv(x2, y2, x, y);
          sum += deriv * row_in[x];
        }
      }
      row[x2] = sum;
    }
  }
}

// Element-wise multiplies two images.
template <typename T>
Status ElwiseMul(const Plane<T>& image1, const Plane<T>& image2,
                 Plane<T>* out) {
  const size_t xsize = image1.xsize();
  const size_t ysize = image1.ysize();
  JXL_ENSURE(xsize == image2.xsize());
  JXL_ENSURE(ysize == image2.ysize());
  JXL_ENSURE(xsize == out->xsize());
  JXL_ENSURE(ysize == out->ysize());
  for (size_t y = 0; y < ysize; ++y) {
    const T* const JXL_RESTRICT row1 = image1.Row(y);
    const T* const JXL_RESTRICT row2 = image2.Row(y);
    T* const JXL_RESTRICT row_out = out->Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      row_out[x] = row1[x] * row2[x];
    }
  }
  return true;
}

// Element-wise divides two images.
template <typename T>
Status ElwiseDiv(const Plane<T>& image1, const Plane<T>& image2,
                 Plane<T>* out) {
  const size_t xsize = image1.xsize();
  const size_t ysize = image1.ysize();
  JXL_ENSURE(xsize == image2.xsize());
  JXL_ENSURE(ysize == image2.ysize());
  JXL_ENSURE(xsize == out->xsize());
  JXL_ENSURE(ysize == out->ysize());
  for (size_t y = 0; y < ysize; ++y) {
    const T* const JXL_RESTRICT row1 = image1.Row(y);
    const T* const JXL_RESTRICT row2 = image2.Row(y);
    T* const JXL_RESTRICT row_out = out->Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      row_out[x] = row1[x] / row2[x];
    }
  }
  return true;
}

void ReduceRinging(const ImageF& initial, const ImageF& mask, ImageF& down) {
  int64_t xsize2 = down.xsize();
  int64_t ysize2 = down.ysize();

  for (size_t y = 0; y < down.ysize(); y++) {
    const float* row_mask = mask.Row(y);
    float* row_out = down.Row(y);
    for (size_t x = 0; x < down.xsize(); x++) {
      float v = down.Row(y)[x];
      float min = initial.Row(y)[x];
      float max = initial.Row(y)[x];
      for (int64_t yi = -1; yi < 2; yi++) {
        for (int64_t xi = -1; xi < 2; xi++) {
          int64_t x2 = static_cast<int64_t>(x) + xi;
          int64_t y2 = static_cast<int64_t>(y) + yi;
          if (x2 < 0 || y2 < 0 || x2 >= xsize2 || y2 >= ysize2) continue;
          min = std::min<float>(min, initial.Row(y2)[x2]);
          max = std::max<float>(max, initial.Row(y2)[x2]);
        }
      }

      row_out[x] = v;

      // Clamp the pixel within the value  of a small area to prevent ringning.
      // The mask determines how much to clamp, clamp more to reduce more
      // ringing in smooth areas, clamp less in noisy areas to get more
      // sharpness. Higher mask_multiplier gives less clamping, so less
      // ringing reduction.
      const constexpr float mask_multiplier = 2;
      float a = row_mask[x] * mask_multiplier;
      float clip_min = min - a;
      float clip_max = max + a;
      if (row_out[x] < clip_min) row_out[x] = clip_min;
      if (row_out[x] > clip_max) row_out[x] = clip_max;
    }
  }
}

// TODO(lode): move this to a separate file enc_downsample.cc
Status DownsampleImage2_Iterative(const ImageF& orig, ImageF* output) {
  int64_t xsize = orig.xsize();
  int64_t ysize = orig.ysize();
  int64_t xsize2 = DivCeil(orig.xsize(), 2);
  int64_t ysize2 = DivCeil(orig.ysize(), 2);
  JxlMemoryManager* memory_manager = orig.memory_manager();

  JXL_ASSIGN_OR_RETURN(ImageF box_downsample,
                       ImageF::Create(memory_manager, xsize, ysize));
  JXL_RETURN_IF_ERROR(CopyImageTo(orig, &box_downsample));
  JXL_ASSIGN_OR_RETURN(box_downsample, DownsampleImage(box_downsample, 2));
  JXL_ASSIGN_OR_RETURN(ImageF mask,
                       ImageF::Create(memory_manager, box_downsample.xsize(),
                                      box_downsample.ysize()));
  CreateMask(box_downsample, mask);

  JXL_RETURN_IF_ERROR(output->ShrinkTo(xsize2, ysize2));

  // Initial result image using the sharper downsampling.
  // Allocate extra space to avoid a reallocation when padding.
  JXL_ASSIGN_OR_RETURN(
      ImageF initial,
      ImageF::Create(memory_manager, DivCeil(orig.xsize(), 2) + kBlockDim,
                     DivCeil(orig.ysize(), 2) + kBlockDim));
  JXL_RETURN_IF_ERROR(initial.ShrinkTo(initial.xsize() - kBlockDim,
                                       initial.ysize() - kBlockDim));
  JXL_RETURN_IF_ERROR(DownsampleImage2_Sharper(orig, &initial));

  JXL_ASSIGN_OR_RETURN(
      ImageF down,
      ImageF::Create(memory_manager, initial.xsize(), initial.ysize()));
  JXL_RETURN_IF_ERROR(CopyImageTo(initial, &down));
  JXL_ASSIGN_OR_RETURN(ImageF up, ImageF::Create(memory_manager, xsize, ysize));
  JXL_ASSIGN_OR_RETURN(ImageF corr,
                       ImageF::Create(memory_manager, xsize, ysize));
  JXL_ASSIGN_OR_RETURN(ImageF corr2,
                       ImageF::Create(memory_manager, xsize2, ysize2));

  // In the weights map, relatively higher values will allow less ringing but
  // also less sharpness. With all constant values, it optimizes equally
  // everywhere. Even in this case, the weights2 computed from
  // this is still used and differs at the borders of the image.
  // TODO(lode): Make use of the weights field for anti-ringing and clamping,
  // the values are all set to 1 for now, but it is intended to be used for
  // reducing ringing based on the mask, and taking clamping into account.
  JXL_ASSIGN_OR_RETURN(ImageF weights,
                       ImageF::Create(memory_manager, xsize, ysize));
  for (size_t y = 0; y < weights.ysize(); y++) {
    auto* row = weights.Row(y);
    for (size_t x = 0; x < weights.xsize(); x++) {
      row[x] = 1;
    }
  }
  JXL_ASSIGN_OR_RETURN(ImageF weights2,
                       ImageF::Create(memory_manager, xsize2, ysize2));
  AntiUpsample(weights, &weights2);

  const size_t num_it = 3;
  for (size_t it = 0; it < num_it; ++it) {
    UpsampleImage(down, &up);
    JXL_ASSIGN_OR_RETURN(corr, LinComb<float>(1, orig, -1, up));
    JXL_RETURN_IF_ERROR(ElwiseMul(corr, weights, &corr));
    AntiUpsample(corr, &corr2);
    JXL_RETURN_IF_ERROR(ElwiseDiv(corr2, weights2, &corr2));

    JXL_ASSIGN_OR_RETURN(down, LinComb<float>(1, down, 1, corr2));
  }

  ReduceRinging(initial, mask, down);

  // can't just use CopyImage, because the output image was prepared with
  // padding.
  for (size_t y = 0; y < down.ysize(); y++) {
    for (size_t x = 0; x < down.xsize(); x++) {
      float v = down.Row(y)[x];
      output->Row(y)[x] = v;
    }
  }
  return true;
}

}  // namespace

Status DownsampleImage2_Iterative(Image3F* opsin) {
  JxlMemoryManager* memory_manager = opsin->memory_manager();
  // Allocate extra space to avoid a reallocation when padding.
  JXL_ASSIGN_OR_RETURN(
      Image3F downsampled,
      Image3F::Create(memory_manager, DivCeil(opsin->xsize(), 2) + kBlockDim,
                      DivCeil(opsin->ysize(), 2) + kBlockDim));
  JXL_RETURN_IF_ERROR(downsampled.ShrinkTo(downsampled.xsize() - kBlockDim,
                                           downsampled.ysize() - kBlockDim));

  JXL_ASSIGN_OR_RETURN(
      Image3F rgb,
      Image3F::Create(memory_manager, opsin->xsize(), opsin->ysize()));
  OpsinParams opsin_params;  // TODO(user): use the ones that are actually used
  opsin_params.Init(kDefaultIntensityTarget);
  JXL_RETURN_IF_ERROR(
      OpsinToLinear(*opsin, Rect(rgb), nullptr, &rgb, opsin_params));

  JXL_ASSIGN_OR_RETURN(
      ImageF mask,
      ImageF::Create(memory_manager, opsin->xsize(), opsin->ysize()));
  ButteraugliParams butter_params;
  JXL_ASSIGN_OR_RETURN(std::unique_ptr<ButteraugliComparator> butter,
                       ButteraugliComparator::Make(rgb, butter_params));
  JXL_RETURN_IF_ERROR(butter->Mask(&mask));
  JXL_ASSIGN_OR_RETURN(
      ImageF mask_fuzzy,
      ImageF::Create(memory_manager, opsin->xsize(), opsin->ysize()));

  for (size_t c = 0; c < 3; c++) {
    JXL_RETURN_IF_ERROR(
        DownsampleImage2_Iterative(opsin->Plane(c), &downsampled.Plane(c)));
  }
  *opsin = std::move(downsampled);
  return true;
}

StatusOr<Image3F> ReconstructImage(
    const FrameHeader& orig_frame_header, const PassesSharedState& shared,
    const std::vector<std::unique_ptr<ACImage>>& coeffs, ThreadPool* pool) {
  const FrameDimensions& frame_dim = shared.frame_dim;
  JxlMemoryManager* memory_manager = shared.memory_manager;

  FrameHeader frame_header = orig_frame_header;
  frame_header.UpdateFlag(shared.image_features.patches.HasAny(),
                          FrameHeader::kPatches);
  frame_header.UpdateFlag(shared.image_features.splines.HasAny(),
                          FrameHeader::kSplines);
  frame_header.color_transform = ColorTransform::kNone;

  auto metadata = jxl::make_unique<CodecMetadata>();
  *metadata = *frame_header.nonserialized_metadata;
  metadata->m.extra_channel_info.clear();
  metadata->m.num_extra_channels = metadata->m.extra_channel_info.size();
  frame_header.nonserialized_metadata = metadata.get();
  frame_header.extra_channel_upsampling.clear();

  const bool is_gray = shared.metadata->m.color_encoding.IsGray();
  auto dec_state = jxl::make_unique<PassesDecoderState>(memory_manager);
  JXL_RETURN_IF_ERROR(
      dec_state->output_encoding_info.SetFromMetadata(*shared.metadata));
  JXL_RETURN_IF_ERROR(dec_state->output_encoding_info.MaybeSetColorEncoding(
      ColorEncoding::LinearSRGB(is_gray)));
  dec_state->shared = &shared;
  JXL_RETURN_IF_ERROR(dec_state->Init(frame_header));

  ImageBundle decoded(memory_manager, &shared.metadata->m);
  decoded.origin = frame_header.frame_origin;
  JXL_ASSIGN_OR_RETURN(
      Image3F tmp,
      Image3F::Create(memory_manager, frame_dim.xsize, frame_dim.ysize));
  JXL_RETURN_IF_ERROR(decoded.SetFromImage(
      std::move(tmp), dec_state->output_encoding_info.color_encoding));

  PassesDecoderState::PipelineOptions options;
  options.use_slow_render_pipeline = false;
  options.coalescing = false;
  options.render_spotcolors = false;
  options.render_noise = true;

  JXL_RETURN_IF_ERROR(dec_state->PreparePipeline(
      frame_header, &shared.metadata->m, &decoded, options));

  AlignedArray<GroupDecCache> group_dec_caches;
  const auto allocate_storage = [&](const size_t num_threads) -> Status {
    JXL_RETURN_IF_ERROR(
        dec_state->render_pipeline->PrepareForThreads(num_threads,
                                                      /*use_group_ids=*/false));
    JXL_ASSIGN_OR_RETURN(group_dec_caches, AlignedArray<GroupDecCache>::Create(
                                               memory_manager, num_threads));
    return true;
  };
  const auto process_group = [&](const uint32_t group_index,
                                 const size_t thread) -> Status {
    if (frame_header.loop_filter.epf_iters > 0) {
      JXL_RETURN_IF_ERROR(ComputeSigma(frame_header.loop_filter,
                                       frame_dim.BlockGroupRect(group_index),
                                       dec_state.get()));
    }
    RenderPipelineInput input =
        dec_state->render_pipeline->GetInputBuffers(group_index, thread);
    JXL_RETURN_IF_ERROR(DecodeGroupForRoundtrip(
        frame_header, coeffs, group_index, dec_state.get(),
        &group_dec_caches[thread], thread, input, nullptr, nullptr));
    if ((frame_header.flags & FrameHeader::kNoise) != 0) {
      PrepareNoiseInput(*dec_state, shared.frame_dim, frame_header, group_index,
                        thread);
    }
    JXL_RETURN_IF_ERROR(input.Done());
    return true;
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, frame_dim.num_groups, allocate_storage,
                                process_group, "ReconstructImage"));
  return std::move(*decoded.color());
}

float ComputeBlockL2Distance(const Image3F& a, const Image3F& b,
                             const ImageF& mask1x1, size_t by, size_t bx) {
  Rect rect(bx * kBlockDim, by * kBlockDim, kBlockDim, kBlockDim, a.xsize(),
            a.ysize());
  const size_t xsize = rect.xsize();
  const size_t ysize = rect.ysize();
  float err2_masked[3] = {0.0f};
  float err2_unmasked[3] = {0.0f};
  float sum_mask2 = 0.0f;
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
    const float* row_mask = rect.ConstRow(mask1x1, y);
    for (size_t x = 0; x < xsize; ++x) {
      float mask = row_mask[x];
      float mask2 = mask * mask;
      sum_mask2 += mask2;
      for (int i = 0; i < 3; ++i) {
        float diff = row_a[i][x] - row_b[i][x];
        float diff2 = diff * diff;
        err2_masked[i] += mask2 * diff2;
        err2_unmasked[i] += diff2;
      }
    }
  }
  const float mean_mask2 =
      sum_mask2 / static_cast<float>(xsize * ysize);
  constexpr float kUnmaskedMix = 0.22f;
  float err2[3];
  for (int i = 0; i < 3; ++i) {
    err2[i] = (1.0f - kUnmaskedMix) * err2_masked[i] +
              kUnmaskedMix * mean_mask2 * err2_unmasked[i];
  }
  static const double kW[] = {
      12.339445295782363,
      1.0,
      0.2,
  };
  float retval = kW[0] * err2[0] + kW[1] * err2[1] + kW[2] * err2[2];
  return retval;
}

Status ComputeARHeuristics(const FrameHeader& frame_header,
                           PassesEncoderState* enc_state,
                           const Image3F& orig_opsin, const Rect& rect,
                           ThreadPool* pool) {
  const CompressParams& cparams = enc_state->cparams;
  PassesSharedState& shared = enc_state->shared;
  const FrameDimensions& frame_dim = shared.frame_dim;
  const ImageF& initial_quant_masking1x1 = enc_state->initial_quant_masking1x1;
  ImageB& epf_sharpness = shared.epf_sharpness;
  JxlMemoryManager* memory_manager = enc_state->memory_manager();

  float clamped_butteraugli = std::min(5.0f, cparams.butteraugli_distance);
  const bool ar_enable_rdo =
      cparams.speed_tier <= SpeedTier::kSquirrel &&
      !cparams.disable_perceptual_optimizations &&
      cparams.butteraugli_distance >= 0.35f &&
      cparams.butteraugli_distance < 3.5f && cparams.resampling <= 1 &&
      cparams.ec_resampling <= 1 && cparams.gaborish != Override::kOff &&
      cparams.epf != 3 &&
      enc_state->progressive_splitter.GetNumPasses() == 1;
  if (frame_header.loop_filter.epf_iters == 0 && ar_enable_rdo) {
    FillPlane(static_cast<uint8_t>(0), &epf_sharpness, Rect(epf_sharpness));
    return true;
  }
  if (cparams.butteraugli_distance < kMinButteraugliForDynamicAR ||
      cparams.speed_tier > SpeedTier::kWombat ||
      frame_header.loop_filter.epf_iters == 0) {
    FillPlane(static_cast<uint8_t>(4), &epf_sharpness, Rect(epf_sharpness));
    return true;
  }

  std::vector<uint8_t> epf_steps;
  if (cparams.butteraugli_distance > 4.5f) {
    epf_steps.push_back(0);
    epf_steps.push_back(4);
  } else {
    epf_steps.push_back(0);
    epf_steps.push_back(2);
    epf_steps.push_back(7);
  }
  static const int kNumEPFVals = 8;
  size_t epf_steps_lut[kNumEPFVals] = {0};
  {
    for (size_t i = 0; i < epf_steps.size(); ++i) {
      epf_steps_lut[epf_steps[i]] = i;
    }
  }
  std::array<ImageF, kNumEPFVals> error_images;
  for (uint8_t val : epf_steps) {
    FillPlane(val, &epf_sharpness, Rect(epf_sharpness));
    JXL_ASSIGN_OR_RETURN(
        Image3F decoded,
        ReconstructImage(frame_header, shared, enc_state->coeffs, pool));
    JXL_ASSIGN_OR_RETURN(error_images[val],
                         ImageF::Create(memory_manager, frame_dim.xsize_blocks,
                                        frame_dim.ysize_blocks));
    for (size_t by = 0; by < frame_dim.ysize_blocks; by++) {
      float* error_row = error_images[val].Row(by);
      for (size_t bx = 0; bx < frame_dim.xsize_blocks; bx++) {
        error_row[bx] = ComputeBlockL2Distance(
            orig_opsin, decoded, initial_quant_masking1x1, by, bx);
      }
    }
  }
  double sum_sqrt_d0 = 0.0;
  for (size_t by = 0; by < frame_dim.ysize_blocks; by++) {
    const float* row0 = error_images[0].ConstRow(by);
    for (size_t bx = 0; bx < frame_dim.xsize_blocks; bx++) {
      sum_sqrt_d0 += std::sqrt(row0[bx]);
    }
  }
  const float mean_sqrt_d0 =
      sum_sqrt_d0 /
      std::max<size_t>(1, frame_dim.xsize_blocks * frame_dim.ysize_blocks);
  const float ref_d0 = mean_sqrt_d0 * mean_sqrt_d0;
  constexpr float kLocalRef = 0.84f;
  const float global_ref =
      0.15f / std::pow(clamped_butteraugli, 0.35f);
  std::vector<std::vector<size_t>> histo(9, std::vector<size_t>(kNumEPFVals));
  std::vector<size_t> totals(9, 1);
  static const float kFavorNoSmoothing = 0.99;
  for (size_t by = 0; by < frame_dim.ysize_blocks; by++) {
    uint8_t* JXL_RESTRICT out_row = epf_sharpness.Row(by);
    uint8_t* JXL_RESTRICT prev_row = epf_sharpness.Row(by > 0 ? by - 1 : 0);
    const float* row0 = error_images[0].ConstRow(by);
    for (size_t bx = 0; bx < frame_dim.xsize_blocks; bx++) {
      const float d_ref = kLocalRef * row0[bx] + global_ref * ref_d0;
      uint8_t best_val = 0;
      float best_error = std::numeric_limits<float>::max();
      uint8_t top_val = by > 0 ? prev_row[bx] : 0;
      uint8_t left_val = bx > 0 ? out_row[bx - 1] : 0;
      float top_mul = 0.985f * (top_val == 0 ? kFavorNoSmoothing : 1.0f);
      float left_mul = 0.985f * (left_val == 0 ? kFavorNoSmoothing : 1.0f);
      float top_error =
          error_images[top_val].Row(by)[bx] + d_ref * (top_mul - 1.0f);
      float left_error =
          error_images[left_val].Row(by)[bx] + d_ref * (left_mul - 1.0f);
      for (uint8_t val : epf_steps) {
        float m = (val == 0) ? kFavorNoSmoothing : 1.0f;
        float error = error_images[val].Row(by)[bx] + d_ref * (m - 1.0f);
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
      int context = epf_steps_lut[top_val] * 3 + epf_steps_lut[left_val];
      ++histo[context][out_row[bx]];
      ++totals[context];
    }
  }
  const float c3base = 0.98017198824148288;
  const float c3clamp = 0.85970338919928291;
  const float c3 = std::max(c3clamp, std::pow(c3base, clamped_butteraugli));
  static const float c5 = 0.1087690359555803;
  float mul[3 * 3 * 3] = {0};
  for (uint8_t top_val : epf_steps) {
    for (uint8_t left_val : epf_steps) {
      for (uint8_t val : epf_steps) {
        int context = epf_steps_lut[top_val] * 3 + epf_steps_lut[left_val];
        const auto& ctx_histo = histo[context];
        const int mulix = epf_steps_lut[val] + 3 * context;
        const float ratio = static_cast<float>(ctx_histo[val]) /
                            static_cast<float>(totals[context]);
        mul[mulix] =
            1.0f / (1.0f + c5 * std::log1p(ratio) / clamped_butteraugli);
        if (val == top_val && val == left_val) {
          mul[mulix] *= 0.988f;
        } else if (val == top_val || val == left_val) {
          mul[mulix] *= 0.994f;
        }
        if (val == 0) {
          mul[mulix] *= c3;
        }
      }
    }
  }
  for (size_t by = 0; by < frame_dim.ysize_blocks; by++) {
    uint8_t* JXL_RESTRICT out_row = epf_sharpness.Row(by);
    uint8_t* JXL_RESTRICT prev_row = epf_sharpness.Row(by > 0 ? by - 1 : 0);
    const float* row0 = error_images[0].ConstRow(by);
    for (size_t bx = 0; bx < frame_dim.xsize_blocks; bx++) {
      const float d_ref = kLocalRef * row0[bx] + global_ref * ref_d0;
      uint8_t best_val = 0;
      float best_error = std::numeric_limits<float>::max();
      uint8_t top_val = by > 0 ? prev_row[bx] : 0;
      uint8_t left_val = bx > 0 ? out_row[bx - 1] : 0;
      int context = epf_steps_lut[top_val] * 3 + epf_steps_lut[left_val];
      for (uint8_t val : epf_steps) {
        int mulix = epf_steps_lut[val] + 3 * context;
        float error =
            error_images[val].Row(by)[bx] + d_ref * (mul[mulix] - 1.0f);
        if (error < best_error) {
          best_val = val;
          best_error = error;
        }
      }
      out_row[bx] = best_val;
    }
  }

  return true;
}

Status LossyFrameHeuristics(const FrameHeader& frame_header,
                            PassesEncoderState* enc_state,
                            ModularFrameEncoder* modular_frame_encoder,
                            const Image3F* linear, Image3F* opsin,
                            const Rect& rect, const JxlCmsInterface& cms,
                            ThreadPool* pool, AuxOut* aux_out) {
  const CompressParams& cparams = enc_state->cparams;
  const bool streaming_mode = enc_state->streaming_mode;
  const bool initialize_global_state = enc_state->initialize_global_state;
  PassesSharedState& shared = enc_state->shared;
  const FrameDimensions& frame_dim = shared.frame_dim;
  ImageFeatures& image_features = shared.image_features;
  DequantMatrices& matrices = shared.matrices;
  Quantizer& quantizer = shared.quantizer;
  ImageF& initial_quant_masking1x1 = enc_state->initial_quant_masking1x1;
  ImageI& raw_quant_field = shared.raw_quant_field;
  ColorCorrelationMap& cmap = shared.cmap;
  AcStrategyImage& ac_strategy = shared.ac_strategy;
  BlockCtxMap& block_ctx_map = shared.block_ctx_map;
  JxlMemoryManager* memory_manager = enc_state->memory_manager();

  // Find and subtract splines.
  bool override_splines = cparams.custom_splines.HasAny();
  if (override_splines) {
    image_features.splines.SetData(cparams.custom_splines);
  }
  if (!streaming_mode && cparams.speed_tier <= SpeedTier::kSquirrel) {
    if (!override_splines) {
      image_features.splines = FindSplines(*opsin);
    }
    JXL_RETURN_IF_ERROR(image_features.splines.InitializeDrawCache(
        opsin->xsize(), opsin->ysize(), cmap.base()));
    image_features.splines.SubtractFrom(opsin);
  }

  // Find and subtract patches/dots.
  if (!streaming_mode &&
      ApplyOverride(cparams.patches,
                    cparams.speed_tier <= SpeedTier::kSquirrel)) {
    JXL_RETURN_IF_ERROR(
        FindBestPatchDictionary(*opsin, enc_state, cms, pool, aux_out));
    JXL_RETURN_IF_ERROR(
        PatchDictionaryEncoder::SubtractFrom(image_features.patches, opsin));
  }

  const float quant_dc = InitialQuantDC(cparams.butteraugli_distance);

  // TODO(veluca): we can now run all the code from here to FindBestQuantizer
  // (excluded) one rect at a time. Do that.

  // Dependency graph:
  //
  // input: either XYB or input image
  //
  // input image -> XYB [optional]
  // XYB -> initial quant field
  // XYB -> Gaborished XYB
  // Gaborished XYB -> CfL1
  // initial quant field, Gaborished XYB, CfL1 -> ACS
  // initial quant field, ACS, Gaborished XYB -> EPF control field
  // initial quant field -> adjusted initial quant field
  // adjusted initial quant field, ACS -> raw quant field
  // raw quant field, ACS, Gaborished XYB -> CfL2
  //
  // output: Gaborished XYB, CfL, ACS, raw quant field, EPF control field.

  AcStrategyHeuristics acs_heuristics(memory_manager, cparams);
  CfLHeuristics cfl_heuristics(memory_manager);
  ImageF initial_quant_field;
  ImageF initial_quant_masking;
  const bool enable_rdo =
      cparams.speed_tier <= SpeedTier::kSquirrel &&
      !cparams.disable_perceptual_optimizations &&
      cparams.butteraugli_distance >= 0.35f &&
      cparams.butteraugli_distance < 3.5f && cparams.resampling <= 1 &&
      cparams.ec_resampling <= 1 && cparams.gaborish != Override::kOff &&
      cparams.epf != 3 &&
      enc_state->progressive_splitter.GetNumPasses() == 1 &&
      frame_dim.xsize_blocks * frame_dim.ysize_blocks > 64;
  if (!enable_rdo) {
    enc_state->band_qf[1][0] = ImageF();
  }

  // Compute an initial estimate of the quantization field.
  // Call InitialQuantField only in Hare mode or slower. Otherwise, rely
  // on simple heuristics in FindBestAcStrategy, or set a constant for Falcon
  // mode.
  if (cparams.speed_tier > SpeedTier::kHare ||
      cparams.disable_perceptual_optimizations) {
    JXL_ASSIGN_OR_RETURN(initial_quant_field,
                         ImageF::Create(memory_manager, frame_dim.xsize_blocks,
                                        frame_dim.ysize_blocks));
    JXL_ASSIGN_OR_RETURN(initial_quant_masking,
                         ImageF::Create(memory_manager, frame_dim.xsize_blocks,
                                        frame_dim.ysize_blocks));
    float q = 0.79 / cparams.butteraugli_distance;
    FillImage(q, &initial_quant_field);
    float masking = 1.0f / (q + 0.001f);
    FillImage(masking, &initial_quant_masking);
    if (cparams.disable_perceptual_optimizations) {
      JXL_ASSIGN_OR_RETURN(
          initial_quant_masking1x1,
          ImageF::Create(memory_manager, frame_dim.xsize, frame_dim.ysize));
      FillImage(masking, &initial_quant_masking1x1);
    }
    quantizer.ComputeGlobalScaleAndQuant(quant_dc, q, 0);
  } else {
    // Call this here, as it relies on pre-gaborish values.
    float butteraugli_distance_for_iqf = cparams.butteraugli_distance;
    if (!frame_header.loop_filter.gab) {
      butteraugli_distance_for_iqf *= 0.62f;
    }
    JXL_ASSIGN_OR_RETURN(
        initial_quant_field,
        InitialQuantField(butteraugli_distance_for_iqf, *opsin, rect, pool,
                          1.0f, &initial_quant_masking,
                          &initial_quant_masking1x1,
                          enable_rdo ? enc_state->band_qf : nullptr));
    float q = 0.39 / cparams.butteraugli_distance;
    quantizer.ComputeGlobalScaleAndQuant(quant_dc, q, 0);
  }

  // TODO(veluca): do something about animations.

  // Apply inverse-gaborish.
  if (frame_header.loop_filter.gab) {
    // Changing the weight here to 0.99f would help to reduce ringing in
    // generation loss.
    float weight[3] = {
        1.0f,
        1.0f,
        1.0f,
    };
    JXL_RETURN_IF_ERROR(GaborishInverse(opsin, rect, weight, pool));
  }

  if (initialize_global_state) {
    JXL_RETURN_IF_ERROR(FindBestDequantMatrices(
        memory_manager, cparams, frame_dim, enable_rdo, modular_frame_encoder,
        &matrices));
  }

  enc_state->x_qm_multiplier =
      std::pow(1.25f, frame_header.x_qm_scale - 2.0f);
  enc_state->b_qm_multiplier =
      std::pow(1.25f, frame_header.b_qm_scale - 2.0f);

  JXL_RETURN_IF_ERROR(cfl_heuristics.Init(rect));
  JXL_RETURN_IF_ERROR(acs_heuristics.Init(
      *opsin, rect, initial_quant_field, initial_quant_masking,
      initial_quant_masking1x1, &matrices,
      enable_rdo ? enc_state->band_qf : nullptr, enc_state->x_qm_multiplier,
      enc_state->b_qm_multiplier));

  const size_t acs_tile_dim_blocks =
      enable_rdo ? kGroupDimInBlocks : kEncTileDimInBlocks;
  auto process_tile = [&](const uint32_t tid, const size_t thread) -> Status {
    size_t n_enc_tiles = DivCeil(frame_dim.xsize_blocks, acs_tile_dim_blocks);
    size_t tx = tid % n_enc_tiles;
    size_t ty = tid / n_enc_tiles;
    size_t gby0 = ty * acs_tile_dim_blocks;
    size_t gby1 =
        std::min((ty + 1) * acs_tile_dim_blocks, frame_dim.ysize_blocks);
    size_t gbx0 = tx * acs_tile_dim_blocks;
    size_t gbx1 =
        std::min((tx + 1) * acs_tile_dim_blocks, frame_dim.xsize_blocks);
    Rect group_rect(gbx0, gby0, gbx1 - gbx0, gby1 - gby0);

    // For speeds up to Wombat, we only compute the color correlation map
    // once we know the transform type and the quantization map.
    if (cparams.speed_tier <= SpeedTier::kSquirrel) {
      if (enable_rdo) {
        quantizer.SetQuantFieldRect(initial_quant_field, group_rect,
                                    &raw_quant_field);
      }
      for (size_t by0 = gby0; by0 < gby1; by0 += kEncTileDimInBlocks) {
        size_t by1 = std::min(by0 + kEncTileDimInBlocks, gby1);
        for (size_t bx0 = gbx0; bx0 < gbx1; bx0 += kEncTileDimInBlocks) {
          size_t bx1 = std::min(bx0 + kEncTileDimInBlocks, gbx1);
          Rect r(bx0, by0, bx1 - bx0, by1 - by0);
          JXL_RETURN_IF_ERROR(cfl_heuristics.ComputeTile(
              r, *opsin, rect, matrices,
              /*ac_strategy=*/nullptr,
              /*raw_quant_field=*/enable_rdo ? &raw_quant_field : nullptr,
              /*quantizer=*/enable_rdo ? &quantizer : nullptr, /*fast=*/false,
              thread, &cmap,
              /*shrink_towards_zero=*/!enable_rdo ||
                  cparams.noise == Override::kOn));
        }
      }
    }

    // Choose block sizes.
    JXL_RETURN_IF_ERROR(
        acs_heuristics.ProcessRect(group_rect, cmap, &ac_strategy, thread));

    // Always set the initial quant field, so we can compute the CfL map with
    // more accuracy. The initial quant field might change in slower modes, but
    // adjusting the quant field with butteraugli when all the other encoding
    // parameters are fixed is likely a more reliable choice anyway.
    JXL_RETURN_IF_ERROR(
        AdjustQuantField(ac_strategy, group_rect, cparams.butteraugli_distance,
                         &initial_quant_field,
                         enable_rdo ? enc_state->band_qf : nullptr));
    quantizer.SetQuantFieldRect(initial_quant_field, group_rect,
                                &raw_quant_field);

    // Compute a non-default CfL map if we are at Hare speed, or slower.
    if (cparams.speed_tier <= SpeedTier::kHare) {
      for (size_t by0 = gby0; by0 < gby1; by0 += kEncTileDimInBlocks) {
        size_t by1 = std::min(by0 + kEncTileDimInBlocks, gby1);
        for (size_t bx0 = gbx0; bx0 < gbx1; bx0 += kEncTileDimInBlocks) {
          size_t bx1 = std::min(bx0 + kEncTileDimInBlocks, gbx1);
          Rect r(bx0, by0, bx1 - bx0, by1 - by0);
          JXL_RETURN_IF_ERROR(cfl_heuristics.ComputeTile(
              r, *opsin, rect, matrices, &ac_strategy, &raw_quant_field,
              &quantizer, /*fast=*/cparams.speed_tier >= SpeedTier::kWombat,
              thread, &cmap,
              /*shrink_towards_zero=*/!enable_rdo ||
                  cparams.noise == Override::kOn));
        }
      }
    }
    return true;
  };
  size_t num_tiles = DivCeil(frame_dim.xsize_blocks, acs_tile_dim_blocks) *
                     DivCeil(frame_dim.ysize_blocks, acs_tile_dim_blocks);
  const auto prepare = [&](const size_t num_threads) -> Status {
    JXL_RETURN_IF_ERROR(acs_heuristics.PrepareForThreads(num_threads));
    JXL_RETURN_IF_ERROR(cfl_heuristics.PrepareForThreads(num_threads));
    return true;
  };
  JXL_RETURN_IF_ERROR(
      RunOnPool(pool, 0, num_tiles, prepare, process_tile, "Enc Heuristics"));

  JXL_RETURN_IF_ERROR(acs_heuristics.Finalize(frame_dim, ac_strategy, aux_out));

  // Refine quantization levels.
  if (!streaming_mode && !cparams.disable_perceptual_optimizations) {
    ImageB& epf_sharpness = shared.epf_sharpness;
    FillPlane(static_cast<uint8_t>(4), &epf_sharpness, Rect(epf_sharpness));
    JXL_RETURN_IF_ERROR(FindBestQuantizer(frame_header, linear, *opsin,
                                          initial_quant_field, enc_state, cms,
                                          pool, aux_out));
  }

  // Choose a context model that depends on the amount of quantization for AC.
  if (cparams.speed_tier < SpeedTier::kFalcon && initialize_global_state) {
    FindBestBlockEntropyModel(cparams, raw_quant_field, ac_strategy,
                              &block_ctx_map);
  }
  return true;
}

}  // namespace jxl
