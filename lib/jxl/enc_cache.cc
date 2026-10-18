// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_cache.h"

#include <jxl/cms_interface.h>
#include <jxl/memory_manager.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/override.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/common.h"
#include "lib/jxl/compressed_dc.h"
#include "lib/jxl/dct_util.h"
#include "lib/jxl/dec_cache.h"
#include "lib/jxl/dec_frame.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_frame.h"
#include "lib/jxl/enc_group.h"
#include "lib/jxl/enc_heuristics.h"
#include "lib/jxl/enc_modular.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/enc_quant_weights.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/passes_state.h"
#include "lib/jxl/quantizer.h"

namespace jxl {

namespace {

JXL_INLINE int32_t ClampedGradI32(int32_t n, int32_t w, int32_t nw) {
  const int32_t m = std::min(n, w);
  const int32_t M = std::max(n, w);
  return Clamp1(n + w - nw, m, M);
}

JXL_INLINE float ClampedGradF32(float n, float w, float nw) {
  const float m = std::min(n, w);
  const float M = std::max(n, w);
  return Clamp1(n + w - nw, m, M);
}

JXL_INLINE void AddHybridUint420(int32_t signed_res, uint32_t hist[64],
                                 uint64_t* JXL_RESTRICT extra_bits) {
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

double HistogramEntropyBits(const uint32_t hist[64], uint32_t total_n,
                            uint64_t extra_bits) {
  double bits = static_cast<double>(extra_bits);
  if (total_n > 0) {
    bits +=
        static_cast<double>(total_n) * std::log2(static_cast<double>(total_n));
    for (size_t i = 0; i < 64; ++i) {
      if (hist[i] > 0) {
        bits -= static_cast<double>(hist[i]) *
                std::log2(static_cast<double>(hist[i]));
      }
    }
  }
  return bits;
}

// Maximum |q_val| in EvalChromaDcChannelCost when 16-bit Modular buffers are
// used. QuantizeDcResidualRdo in AddVarDCTDC can shift a block's quantized DC
// value by at most |diff_adj| + |best_m - abs_s| <= 0.15 + 2.0 = 2.15 from
// raw_target, while q_val is within 1.0 of raw_target, so |q_out - q_val| <= 4.
// 32760 leaves 7 steps of margin below int16_t max (32767).
constexpr int32_t kMaxSafeAbsQ = 32760;

struct DcChannelRdCost {
  double bits = 0.0;
  double dist = 0.0;
  bool valid = true;
  double total(double lambda_dist) const { return bits + lambda_dist * dist; }
};

DcChannelRdCost EvalChromaDcChannelCost(
    const FrameDimensions& frame_dim, const Image3F& dc_src,
    const Image3F& dc_orig, const ImageI& quant_y, size_t c, float inv_dc_c,
    float half_y_step, float k_ratio, float b_y_comp, int32_t max_abs_q) {
  const float y_comp = (c == 2) ? b_y_comp : 0.0f;
  const float yk_eff = half_y_step * (k_ratio - y_comp);
  const float inv_f = inv_dc_c * 2.0f;
  const float dz = (c == 0) ? 0.64f : 0.60f;
  uint32_t hist[64] = {0};
  uint64_t extra_bits = 0;
  uint32_t total_n = 0;
  double dist = 0.0;
  std::vector<int32_t> q_prev(frame_dim.group_dim, 0);
  std::vector<int32_t> q_cur(frame_dim.group_dim, 0);

  for (size_t group_index = 0; group_index < frame_dim.num_dc_groups;
       ++group_index) {
    const Rect r = frame_dim.DCGroupRect(group_index);
    for (size_t y = 0; y < r.ysize(); ++y) {
      const float* JXL_RESTRICT rc = r.ConstPlaneRow(dc_src, c, y);
      const float* JXL_RESTRICT rc_orig = r.ConstPlaneRow(dc_orig, c, y);
      const float* JXL_RESTRICT ry = r.ConstPlaneRow(dc_orig, 1, y);
      const int32_t* JXL_RESTRICT qy = r.ConstRow(quant_y, y);
      for (size_t x = 0; x < r.xsize(); ++x) {
        const int32_t left = (x > 0) ? q_cur[x - 1] : (y > 0 ? q_prev[x] : 0);
        const int32_t top = (y > 0) ? q_prev[x] : left;
        const int32_t tl = (x > 0 && y > 0) ? q_prev[x - 1] : left;
        const int32_t guess = ClampedGradI32(top, left, tl);
        const float pred_y_cfl =
            y_comp * ry[x] + static_cast<float>(qy[x]) * yk_eff;
        const float raw_target = (rc[x] - pred_y_cfl) * inv_f;
        const float orig_target = (rc_orig[x] - pred_y_cfl) * inv_f;
        const float s = raw_target - static_cast<float>(guess);
        const float abs_s = std::abs(s);
        int32_t res = 0;
        if (abs_s >= dz) {
          if (abs_s < 2.0f) {
            res = static_cast<int32_t>(std::lround(s));
          } else {
            res = static_cast<int32_t>(std::lround(s * 0.5f)) * 2;
          }
        }
        const int32_t q_val = guess + res;
        if (max_abs_q < std::numeric_limits<int32_t>::max() &&
            (std::abs(raw_target) > static_cast<float>(max_abs_q) ||
             q_val > max_abs_q || q_val < -max_abs_q)) {
          DcChannelRdCost out;
          out.valid = false;
          return out;
        }
        q_cur[x] = q_val;
        const float err = orig_target - static_cast<float>(q_val);
        dist += static_cast<double>(err * err);
        AddHybridUint420(res, hist, &extra_bits);
        total_n++;
      }
      std::swap(q_prev, q_cur);
    }
  }
  DcChannelRdCost out;
  out.bits = HistogramEntropyBits(hist, total_n, extra_bits);
  out.dist = dist;
  return out;
}

void ComputeClosedLoopDcCfL(const FrameDimensions& frame_dim, const Image3F& dc,
                            const ImageI& quant_y, bool has_dc_rdo,
                            PassesSharedState* shared) {
  const float inv_dc_x = shared->quantizer.GetInvDcStep(0);
  const float inv_dc_b = shared->quantizer.GetInvDcStep(2);
  const float half_y_step = shared->quantizer.GetDcStep(1) * 0.5f;
  const float base_b = shared->cmap.base().GetBaseCorrelationB();
  const float color_factor = shared->cmap.base().GetColorFactor();
  const float b_y_comp = has_dc_rdo ? 0.75f : 0.0f;
  const bool clamp_i16 = shared->metadata != nullptr &&
                         shared->metadata->m.modular_16_bit_buffer_sufficient;
  const int32_t max_abs_q =
      clamp_i16 ? kMaxSafeAbsQ : std::numeric_limits<int32_t>::max();

  // Check if AC CfL already modified the global ColorCorrelation header.
  const bool header_already_paid =
      color_factor != kDefaultColorFactor ||
      shared->cmap.base().GetBaseCorrelationX() != 0.0f ||
      base_b != jxl::cms::kYToBRatio;
  const double kHeaderBits = header_already_paid ? 0.0 : 50.0;

  double num_x = 0.0;
  double den_x = 1e-5;
  double num_b = 0.0;
  double den_b = 1e-5;
  uint64_t l1_weights_x[256] = {0};
  uint64_t l1_weights_b[256] = {0};
  uint64_t total_w_x = 0;
  uint64_t total_w_b = 0;

  for (size_t group_index = 0; group_index < frame_dim.num_dc_groups;
       ++group_index) {
    const Rect r = frame_dim.DCGroupRect(group_index);
    for (size_t y = 0; y < r.ysize(); ++y) {
      const float* JXL_RESTRICT rx = r.ConstPlaneRow(dc, 0, y);
      const float* JXL_RESTRICT rx_p =
          (y > 0) ? r.ConstPlaneRow(dc, 0, y - 1) : nullptr;
      const float* JXL_RESTRICT ry = r.ConstPlaneRow(dc, 1, y);
      const float* JXL_RESTRICT ry_p =
          (y > 0) ? r.ConstPlaneRow(dc, 1, y - 1) : nullptr;
      const float* JXL_RESTRICT rb = r.ConstPlaneRow(dc, 2, y);
      const float* JXL_RESTRICT rb_p =
          (y > 0) ? r.ConstPlaneRow(dc, 2, y - 1) : nullptr;
      const int32_t* JXL_RESTRICT qy = r.ConstRow(quant_y, y);
      const int32_t* JXL_RESTRICT qy_p =
          (y > 0) ? r.ConstRow(quant_y, y - 1) : nullptr;
      for (size_t x = 0; x < r.xsize(); ++x) {
        const int32_t qy_l = (x > 0) ? qy[x - 1] : (y > 0 ? qy_p[x] : 0);
        const int32_t qy_t = (y > 0) ? qy_p[x] : qy_l;
        const int32_t qy_tl = (x > 0 && y > 0) ? qy_p[x - 1] : qy_l;
        const int32_t dq_y = qy[x] - ClampedGradI32(qy_t, qy_l, qy_tl);
        if (dq_y == 0) continue;
        const float dy = static_cast<float>(dq_y) * half_y_step;

        const float rx_l = (x > 0) ? rx[x - 1] : (y > 0 ? rx_p[x] : 0.0f);
        const float rx_t = (y > 0) ? rx_p[x] : rx_l;
        const float rx_tl = (x > 0 && y > 0) ? rx_p[x - 1] : rx_l;
        const float dx = rx[x] - ClampedGradF32(rx_t, rx_l, rx_tl);

        const float b_cur = rb[x] - b_y_comp * ry[x];
        const float b_l =
            (x > 0) ? (rb[x - 1] - b_y_comp * ry[x - 1])
                    : (y > 0 ? (rb_p[x] - b_y_comp * ry_p[x]) : 0.0f);
        const float b_t = (y > 0) ? (rb_p[x] - b_y_comp * ry_p[x]) : b_l;
        const float b_tl =
            (x > 0 && y > 0) ? (rb_p[x - 1] - b_y_comp * ry_p[x - 1]) : b_l;
        const float db_raw = b_cur - ClampedGradF32(b_t, b_l, b_tl);
        const float db = db_raw - (base_b - b_y_comp) * dy;

        const float w =
            1.0f / std::sqrt(1.0f + 0.04f * static_cast<float>(dq_y * dq_y));
        num_x += w * dy * dx;
        den_x += w * dy * dy;
        num_b += w * dy * db;
        den_b += w * dy * dy;

        const uint64_t abs_dq = static_cast<uint64_t>(std::abs(dq_y));
        const int32_t kx =
            Clamp1<int32_t>(std::lround((dx / dy) * color_factor), -128, 127);
        const int32_t kb =
            Clamp1<int32_t>(std::lround((db / dy) * color_factor), -128, 127);
        l1_weights_x[kx + 128] += abs_dq;
        l1_weights_b[kb + 128] += abs_dq;
        total_w_x += abs_dq;
        total_w_b += abs_dq;
      }
    }
  }

  auto find_l1_median = [](const uint64_t weights[256],
                           uint64_t total_w) -> int32_t {
    if (total_w == 0) return 0;
    const uint64_t half = (total_w + 1) / 2;
    uint64_t acc = 0;
    for (int i = 0; i < 256; ++i) {
      acc += weights[i];
      if (acc >= half) return i - 128;
    }
    return 0;
  };

  const int32_t est_x_l2 =
      Clamp1<int32_t>(std::lround((num_x / den_x) * color_factor), -128, 127);
  const int32_t est_b_l2 =
      Clamp1<int32_t>(std::lround((num_b / den_b) * color_factor), -128, 127);
  const int32_t est_x_l1 = find_l1_median(l1_weights_x, total_w_x);
  const int32_t est_b_l1 = find_l1_median(l1_weights_b, total_w_b);

  constexpr double kLambdaX = 3.5;
  constexpr double kLambdaB = 4.5;
  const size_t num_blocks = frame_dim.xsize_blocks * frame_dim.ysize_blocks;
  const double min_ch_gain = 4.0 + 0.002 * static_cast<double>(num_blocks);

  auto search_channel = [&](size_t c, float inv_dc, int32_t seed_l2,
                            int32_t seed_l1, double lambda_d,
                            double* out_gain) -> int32_t {
    auto eval = [&](int32_t idx) -> DcChannelRdCost {
      const float ratio = (c == 0) ? shared->cmap.base().YtoXRatio(idx)
                                   : shared->cmap.base().YtoBRatio(idx);
      return EvalChromaDcChannelCost(frame_dim, dc, dc, quant_y, c, inv_dc,
                                     half_y_step, ratio, b_y_comp, max_abs_q);
    };
    const DcChannelRdCost base = eval(0);
    if (!base.valid) {
      *out_gain = 0.0;
      return 0;
    }
    const double base_score = base.total(lambda_d);
    double best_score = base_score;
    int32_t best_idx = 0;
    bool seen[256] = {false};
    seen[128] = true;
    auto try_cand = [&](int32_t raw_cand) {
      const int32_t cand = Clamp1<int32_t>(raw_cand, -128, 127);
      if (seen[cand + 128]) return;
      seen[cand + 128] = true;
      const DcChannelRdCost cost = eval(cand);
      if (!cost.valid || cost.dist > base.dist * 1.03) return;
      const double score = cost.total(lambda_d);
      if (score < best_score) {
        best_score = score;
        best_idx = cand;
      }
    };
    try_cand(seed_l2 / 2);
    try_cand((seed_l2 * 3) / 4);
    for (int32_t d = -8; d <= 8; ++d) try_cand(seed_l2 + d);
    for (int32_t d = -6; d <= 6; ++d) try_cand(seed_l1 + d);
    const double gain = base_score - best_score;
    if (gain < min_ch_gain) {
      *out_gain = 0.0;
      return 0;
    }
    *out_gain = gain;
    return best_idx;
  };

  double gain_x = 0.0;
  double gain_b = 0.0;
  const int32_t best_x =
      search_channel(0, inv_dc_x, est_x_l2, est_x_l1, kLambdaX, &gain_x);
  const int32_t best_b =
      search_channel(2, inv_dc_b, est_b_l2, est_b_l1, kLambdaB, &gain_b);

  if (gain_x + gain_b > kHeaderBits + 4.0) {
    shared->cmap.base().SetYToXDC(best_x);
    shared->cmap.base().SetYToBDC(best_b);
  }
}

}  // namespace

Status ComputeACMetadata(ThreadPool* pool, PassesEncoderState* enc_state,
                         ModularFrameEncoder* modular_frame_encoder) {
  PassesSharedState& shared = enc_state->shared;
  auto compute_ac_meta = [&](int group_index, int /* thread */) -> Status {
    const Rect r = shared.frame_dim.DCGroupRect(group_index);
    int modular_group_index = group_index;
    if (enc_state->streaming_mode) {
      JXL_ENSURE(group_index == 0);
      modular_group_index = enc_state->dc_group_index;
    }
    JXL_RETURN_IF_ERROR(modular_frame_encoder->AddACMetadata(
        r, modular_group_index,
        /*jpeg_transcode=*/false, enc_state));
    return true;
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, shared.frame_dim.num_dc_groups,
                                ThreadPool::NoInit, compute_ac_meta,
                                "Compute AC Metadata"));
  return true;
}

Status InitializePassesEncoder(FrameHeader& frame_header,
                               const Image3F& opsin, const Rect& rect,
                               const JxlCmsInterface& cms, ThreadPool* pool,
                               PassesEncoderState* enc_state,
                               ModularFrameEncoder* modular_frame_encoder,
                               AuxOut* aux_out) {
  PassesSharedState& JXL_RESTRICT shared = enc_state->shared;
  JxlMemoryManager* memory_manager = enc_state->memory_manager();

  enc_state->x_qm_multiplier = std::pow(1.25f, frame_header.x_qm_scale - 2.0f);
  enc_state->b_qm_multiplier = std::pow(1.25f, frame_header.b_qm_scale - 2.0f);

  if (enc_state->coeffs.size() < frame_header.passes.num_passes) {
    enc_state->coeffs.reserve(frame_header.passes.num_passes);
    for (size_t i = enc_state->coeffs.size();
         i < frame_header.passes.num_passes; i++) {
      // Allocate enough coefficients for each group on every row.
      JXL_ASSIGN_OR_RETURN(
          std::unique_ptr<ACImageT<int32_t>> coeffs,
          ACImageT<int32_t>::Make(memory_manager, kGroupDim * kGroupDim,
                                  shared.frame_dim.num_groups));
      enc_state->coeffs.emplace_back(std::move(coeffs));
    }
  }
  while (enc_state->coeffs.size() > frame_header.passes.num_passes) {
    enc_state->coeffs.pop_back();
  }

  if (enc_state->initialize_global_state) {
    float scale =
        shared.quantizer.ScaleGlobalScale(enc_state->cparams.quant_ac_rescale);
    JXL_RETURN_IF_ERROR(
        DequantMatricesScaleDC(memory_manager, &shared.matrices, scale));
    shared.quantizer.RecomputeFromGlobalScale();
  }

  if (enc_state->token_cost_model != nullptr) {
    JXL_RETURN_IF_ERROR(BuildPass2FrozenTokenCostModel(
        frame_header, opsin, rect, pool, enc_state));
    JXL_RETURN_IF_ERROR(RefineQuantFieldClosedLoop(frame_header, opsin, rect,
                                                   pool, enc_state));
  }

  JXL_ASSIGN_OR_RETURN(
      Image3F dc, Image3F::Create(memory_manager, shared.frame_dim.xsize_blocks,
                                  shared.frame_dim.ysize_blocks));
  const auto process_group = [&](size_t group_idx, size_t _) -> Status {
    JXL_RETURN_IF_ERROR(
        ComputeCoefficients(group_idx, enc_state, opsin, rect, &dc));
    return true;
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, shared.frame_dim.num_groups,
                                ThreadPool::NoInit, process_group,
                                "Compute coeffs"));

  if (frame_header.flags & FrameHeader::kUseDcFrame) {
    CompressParams cparams = enc_state->cparams;
    cparams.dots = Override::kOff;
    cparams.noise = Override::kOff;
    cparams.patches = Override::kOff;
    cparams.gaborish = Override::kOff;
    cparams.epf = 0;
    cparams.resampling = 1;
    cparams.ec_resampling = 1;
    // The DC frame will have alpha=0. Don't erase its contents.
    cparams.keep_invisible = Override::kOn;
    JXL_ENSURE(cparams.progressive_dc > 0);
    cparams.progressive_dc--;
    // Use kVarDCT in max_error_mode for intermediate progressive DC,
    // and kModular for the smallest DC (first in the bitstream)
    if (cparams.progressive_dc == 0) {
      cparams.modular_mode = true;
      cparams.speed_tier = static_cast<SpeedTier>(
          std::max(static_cast<int>(SpeedTier::kTortoise),
                   static_cast<int>(cparams.speed_tier) - 1));
      cparams.butteraugli_distance =
          std::max(kMinButteraugliDistance * 0.02f,
                   enc_state->cparams.butteraugli_distance * 0.02f);
    } else {
      cparams.max_error_mode = true;
      for (size_t c = 0; c < 3; c++) {
        cparams.max_error[c] = shared.quantizer.MulDC()[c];
      }
      // Guess a distance that produces good initial results.
      cparams.butteraugli_distance =
          std::max(kMinButteraugliDistance,
                   enc_state->cparams.butteraugli_distance * 0.1f);
    }
    auto ib =
        jxl::make_unique<ImageBundle>(memory_manager, &shared.metadata->m);
    // This is a lie - dc is in XYB
    // (but EncodeFrame will skip RGB->XYB conversion anyway)
    JXL_RETURN_IF_ERROR(ib->SetFromImage(
        std::move(dc),
        ColorEncoding::LinearSRGB(shared.metadata->m.color_encoding.IsGray())));
    if (!ib->metadata()->extra_channel_info.empty()) {
      // Add placeholder extra channels to the patch image: dc_level frames do
      // not yet support extra channels, but the codec expects that the amount
      // of extra channels in frames matches that in the metadata of the
      // codestream.
      std::vector<ImageF> extra_channels;
      extra_channels.reserve(ib->metadata()->extra_channel_info.size());
      for (size_t i = 0; i < ib->metadata()->extra_channel_info.size(); i++) {
        JXL_ASSIGN_OR_RETURN(
            ImageF ch,
            ImageF::Create(memory_manager, ib->xsize(), ib->ysize()));
        extra_channels.emplace_back(std::move(ch));
        // Must initialize the image with data to not affect blending with
        // uninitialized memory.
        // TODO(lode): dc_level must copy and use the real extra channels
        // instead.
        ZeroFillImage(&extra_channels.back());
      }
      JXL_RETURN_IF_ERROR(ib->SetExtraChannels(std::move(extra_channels)));
    }
    auto special_frame = jxl::make_unique<BitWriter>(memory_manager);
    FrameInfo dc_frame_info;
    dc_frame_info.frame_type = FrameType::kDCFrame;
    dc_frame_info.dc_level = frame_header.dc_level + 1;
    dc_frame_info.ib_needs_color_transform = false;
    dc_frame_info.save_before_color_transform = true;  // Implicitly true
    AuxOut dc_aux_out;
    JXL_RETURN_IF_ERROR(EncodeFrame(
        memory_manager, cparams, dc_frame_info, shared.metadata, *ib, cms, pool,
        special_frame.get(), aux_out ? &dc_aux_out : nullptr));
    if (aux_out) {
      for (const auto& l : dc_aux_out.layers) {
        aux_out->layer(LayerType::Dc).Assimilate(l);
      }
    }
    const Span<const uint8_t> encoded = special_frame->GetSpan();
    enc_state->special_frames.emplace_back(std::move(special_frame));

    auto decoded =
        jxl::make_unique<ImageBundle>(memory_manager, &shared.metadata->m);
    std::unique_ptr<PassesDecoderState> dec_state =
        jxl::make_unique<PassesDecoderState>(memory_manager);
    JXL_RETURN_IF_ERROR(
        dec_state->output_encoding_info.SetFromMetadata(*shared.metadata));
    const uint8_t* frame_start = encoded.data();
    size_t encoded_size = encoded.size();
    for (int i = 0; i <= cparams.progressive_dc; ++i) {
      JXL_RETURN_IF_ERROR(DecodeFrame(
          dec_state.get(), pool, frame_start, encoded_size,
          /*frame_header=*/nullptr, decoded.get(), *shared.metadata));
      frame_start += decoded->decoded_bytes();
      encoded_size -= decoded->decoded_bytes();
    }
    // TODO(lode): frame_header.dc_level should be equal to
    // dec_state.frame_header.dc_level - 1 here, since above we set
    // dc_frame_info.dc_level = frame_header.dc_level + 1, and
    // dc_frame_info.dc_level is used by EncodeFrame. However, if EncodeFrame
    // outputs multiple frames, this assumption could be wrong.
    const Image3F& dc_frame =
        dec_state->shared->dc_frames[frame_header.dc_level];
    JXL_ASSIGN_OR_RETURN(
        shared.dc_storage,
        Image3F::Create(memory_manager, dc_frame.xsize(), dc_frame.ysize()));
    JXL_RETURN_IF_ERROR(CopyImageTo(dc_frame, &shared.dc_storage));
    ZeroFillImage(&shared.quant_dc);
    shared.dc = &shared.dc_storage;
    JXL_ENSURE(encoded_size == 0);
  } else {
    const bool nl_dc = enc_state->cparams.speed_tier < SpeedTier::kFalcon;
    const bool enable_dc_heuristics =
        !enc_state->streaming_mode && enc_state->initialize_global_state &&
        enc_state->cparams.use_full_image_heuristics &&
        frame_header.chroma_subsampling.Is444() &&
        frame_header.color_transform == ColorTransform::kXYB &&
        !enc_state->cparams.disable_perceptual_optimizations &&
        enc_state->cparams.speed_tier <= SpeedTier::kSquirrel;
    const size_t xsize_blocks = shared.frame_dim.xsize_blocks;
    const size_t ysize_blocks = shared.frame_dim.ysize_blocks;
    const bool has_dc_rdo = enc_state->band_qf[0][3].xsize() >= xsize_blocks &&
                            enc_state->band_qf[0][3].ysize() >= ysize_blocks;

    ImageI quant_y;
    if (enable_dc_heuristics && xsize_blocks >= 2 && ysize_blocks >= 2) {
      JXL_ASSIGN_OR_RETURN(
          quant_y, ImageI::Create(memory_manager, xsize_blocks, ysize_blocks));
      for (size_t group_index = 0;
           group_index < shared.frame_dim.num_dc_groups; ++group_index) {
        const Rect r = shared.frame_dim.DCGroupRect(group_index);
        JXL_RETURN_IF_ERROR(modular_frame_encoder->QuantizeVarDCTYDC(
            frame_header, dc, r, group_index, nl_dc, enc_state, &quant_y));
      }
      if (xsize_blocks * ysize_blocks > 64) {
        ComputeClosedLoopDcCfL(shared.frame_dim, dc, quant_y, has_dc_rdo,
                               &shared);
      }
    }

    const ImageI* precomputed_y_ptr =
        (quant_y.xsize() > 0) ? &quant_y : nullptr;
    auto compute_dc_coeffs = [&](int group_index, int /* thread */) -> Status {
      const Rect r = enc_state->shared.frame_dim.DCGroupRect(group_index);
      int modular_group_index = group_index;
      if (enc_state->streaming_mode) {
        JXL_ENSURE(group_index == 0);
        modular_group_index = enc_state->dc_group_index;
      }
      JXL_RETURN_IF_ERROR(modular_frame_encoder->AddVarDCTDC(
          frame_header, dc, r, modular_group_index, nl_dc, enc_state,
          /*jpeg_transcode=*/false, precomputed_y_ptr));
      return true;
    };
    JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, shared.frame_dim.num_dc_groups,
                                  ThreadPool::NoInit, compute_dc_coeffs,
                                  "Compute DC coeffs"));
    // TODO(veluca): this is only useful in tests and if inspection is enabled.
    if (!(frame_header.flags & FrameHeader::kSkipAdaptiveDCSmoothing)) {
      const size_t xsize_dc = shared.dc_storage.xsize();
      const size_t ysize_dc = shared.dc_storage.ysize();
      const bool has_aq = enc_state->band_qf[1][3].xsize() >= xsize_dc &&
                          enc_state->band_qf[1][3].ysize() >= ysize_dc;
      if (!enc_state->streaming_mode &&
          enc_state->cparams.use_full_image_heuristics &&
          !enc_state->cparams.disable_perceptual_optimizations &&
          enc_state->cparams.speed_tier <= SpeedTier::kSquirrel && has_aq &&
          xsize_dc > 2 && ysize_dc > 2) {
        JXL_ASSIGN_OR_RETURN(
            Image3F smoothed_dc,
            Image3F::Create(memory_manager, xsize_dc, ysize_dc));
        JXL_RETURN_IF_ERROR(CopyImageTo(shared.dc_storage, &smoothed_dc));
        JXL_RETURN_IF_ERROR(AdaptiveDCSmoothing(
            memory_manager, shared.quantizer.MulDC(), &smoothed_dc, pool));
        const float scale_dc = 1.0f / shared.quantizer.inv_quant_dc();
        const float inv_dc_x = kInvDCQuant[0] * scale_dc;
        const float inv_dc_y = kInvDCQuant[1] * scale_dc;
        const float inv_dc_b = kInvDCQuant[2] * scale_dc;
        const float wx = inv_dc_x * inv_dc_x;
        const float wy = inv_dc_y * inv_dc_y;
        const float wb = inv_dc_b * inv_dc_b;
        double sum_log_q = 0.0;
        for (size_t by = 1; by + 1 < ysize_dc; ++by) {
          const float* JXL_RESTRICT q_row =
              enc_state->band_qf[1][3].ConstRow(by);
          for (size_t bx = 1; bx + 1 < xsize_dc; ++bx) {
            sum_log_q += std::log(std::max(1e-4f, q_row[bx]));
          }
        }
        const float q_ref = std::exp(
            sum_log_q / static_cast<double>((xsize_dc - 2) * (ysize_dc - 2)));
        double total_delta = 0.0;
        for (size_t by = 1; by + 1 < ysize_dc; ++by) {
          const float* JXL_RESTRICT rx_t = dc.ConstPlaneRow(0, by);
          const float* JXL_RESTRICT ry_t = dc.ConstPlaneRow(1, by);
          const float* JXL_RESTRICT rb_t = dc.ConstPlaneRow(2, by);
          const float* JXL_RESTRICT rx_m =
              shared.dc_storage.ConstPlaneRow(0, by);
          const float* JXL_RESTRICT ry_m =
              shared.dc_storage.ConstPlaneRow(1, by);
          const float* JXL_RESTRICT rb_m =
              shared.dc_storage.ConstPlaneRow(2, by);
          const float* JXL_RESTRICT rx_s = smoothed_dc.ConstPlaneRow(0, by);
          const float* JXL_RESTRICT ry_s = smoothed_dc.ConstPlaneRow(1, by);
          const float* JXL_RESTRICT rb_s = smoothed_dc.ConstPlaneRow(2, by);
          const float* JXL_RESTRICT q_row =
              has_aq ? enc_state->band_qf[1][3].ConstRow(by) : nullptr;
          for (size_t bx = 1; bx + 1 < xsize_dc; ++bx) {
            const float w_aq =
                has_aq ? Clamp1(q_row[bx] / q_ref, 0.5f, 2.0f) : 1.0f;
            const float dy =
                (ry_s[bx] - ry_m[bx]) * ((ry_s[bx] - ry_t[bx]) +
                                         (ry_m[bx] - ry_t[bx])) *
                wy;
            const float dx =
                (rx_s[bx] - rx_m[bx]) * ((rx_s[bx] - rx_t[bx]) +
                                         (rx_m[bx] - rx_t[bx])) *
                wx;
            const float t_by = rb_t[bx] - ry_t[bx];
            const float m_by = rb_m[bx] - ry_m[bx];
            const float s_by = rb_s[bx] - ry_s[bx];
            const float db = (s_by - m_by) * ((s_by - t_by) + (m_by - t_by)) * wb;
            total_delta +=
                static_cast<double>(w_aq) * (1.0f * dy + 0.35f * dx + 0.25f * db);
          }
        }
        // Incremental U64 header cost for kSkipAdaptiveDCSmoothing (flag=128).
        const uint64_t effective_flags =
            frame_header.flags |
            (shared.image_features.patches.HasAny() ? FrameHeader::kPatches
                                                    : 0) |
            (shared.image_features.splines.HasAny() ? FrameHeader::kSplines
                                                    : 0);
        const double skip_header_bits =
            (effective_flags == 0) ? 8.0 : (effective_flags < 17 ? 4.0 : 0.0);
        const double skip_threshold = 0.25 * skip_header_bits;
        if (total_delta > skip_threshold) {
          frame_header.UpdateFlag(true, FrameHeader::kSkipAdaptiveDCSmoothing);
        } else {
          shared.dc_storage.Swap(smoothed_dc);
        }
      } else {
        JXL_RETURN_IF_ERROR(AdaptiveDCSmoothing(
            memory_manager, shared.quantizer.MulDC(), &shared.dc_storage, pool));
      }
    }
  }
  return true;
}

}  // namespace jxl
