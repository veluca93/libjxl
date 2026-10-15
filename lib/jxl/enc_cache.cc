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
#include <memory>
#include <utility>
#include <vector>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/override.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/common.h"
#include "lib/jxl/compressed_dc.h"
#include "lib/jxl/dct_util.h"
#include "lib/jxl/dec_cache.h"
#include "lib/jxl/dec_frame.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_frame.h"
#include "lib/jxl/enc_group.h"
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

Status InitializePassesEncoder(const FrameHeader& frame_header,
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
    if (enc_state->initialize_global_state &&
        enc_state->cparams.use_full_image_heuristics &&
        frame_header.chroma_subsampling.Is444() &&
        !enc_state->cparams.disable_perceptual_optimizations &&
        enc_state->cparams.speed_tier < SpeedTier::kFalcon &&
        shared.frame_dim.xsize_blocks * shared.frame_dim.ysize_blocks > 64) {
      const size_t xsize = shared.frame_dim.xsize_blocks;
      const size_t ysize = shared.frame_dim.ysize_blocks;
      const float inv_dc_x = shared.quantizer.GetInvDcStep(0);
      const float inv_dc_y = shared.quantizer.GetInvDcStep(1);
      const float inv_dc_b = shared.quantizer.GetInvDcStep(2);
      const float y_step = shared.quantizer.GetDcStep(1);
      auto clamped_grad = [](float n, float w, float nw) -> float {
        const float m = std::min(n, w);
        const float M = std::max(n, w);
        return Clamp1(n + w - nw, m, M);
      };
      double num_x = 0.0;
      double den_x = 1e-5;
      double num_b = 0.0;
      double den_b = 1e-5;
      const float base_b = shared.cmap.base().GetBaseCorrelationB();
      for (size_t y = 1; y < ysize; ++y) {
        const float* JXL_RESTRICT rx = dc.ConstPlaneRow(0, y);
        const float* JXL_RESTRICT rx_p = dc.ConstPlaneRow(0, y - 1);
        const float* JXL_RESTRICT ry = dc.ConstPlaneRow(1, y);
        const float* JXL_RESTRICT ry_p = dc.ConstPlaneRow(1, y - 1);
        const float* JXL_RESTRICT rb = dc.ConstPlaneRow(2, y);
        const float* JXL_RESTRICT rb_p = dc.ConstPlaneRow(2, y - 1);
        for (size_t x = 1; x < xsize; ++x) {
          const float dy =
              ry[x] - clamped_grad(ry_p[x], ry[x - 1], ry_p[x - 1]);
          const float dx =
              rx[x] - clamped_grad(rx_p[x], rx[x - 1], rx_p[x - 1]);
          const float db =
              (rb[x] - base_b * ry[x]) -
              clamped_grad(rb_p[x] - base_b * ry_p[x],
                           rb[x - 1] - base_b * ry[x - 1],
                           rb_p[x - 1] - base_b * ry_p[x - 1]);
          const float q_dy = dy * inv_dc_y;
          const float w = 1.0f / (1.0f + 0.02f * q_dy * q_dy);
          num_x += w * dy * dx;
          den_x += w * dy * dy;
          num_b += w * dy * db;
          den_b += w * dy * dy;
        }
      }
      const int32_t est_x = Clamp1<int32_t>(
          std::lround((num_x / den_x) * shared.cmap.base().GetColorFactor()),
          -128, 127);
      const int32_t est_b = Clamp1<int32_t>(
          std::lround((num_b / den_b) * shared.cmap.base().GetColorFactor()),
          -128, 127);

      const float inv_fy = inv_dc_y * 2.0f;
      const float half_y_step = y_step * 0.5f;
      auto eval_cfl_cost = [&](size_t c, float inv_dc, int32_t idx) -> double {
        const float k = (c == 0) ? shared.cmap.base().YtoXRatio(idx)
                                 : shared.cmap.base().YtoBRatio(idx);
        const float yk = half_y_step * k;
        const float inv_f = inv_dc * 2.0f;
        const float dz = (c == 0) ? 0.65f : 0.61f;
        const float bias1 = (c == 0) ? 0.07f : 0.03f;
        const float bias2 = (c == 0) ? 0.04f : 0.0f;
        const float lambda_dist = (c == 0) ? 4.5f : 3.0f;
        double cost = 0.0;
        for (size_t y = 1; y < ysize; ++y) {
          const float* JXL_RESTRICT rc = dc.ConstPlaneRow(c, y);
          const float* JXL_RESTRICT rc_p = dc.ConstPlaneRow(c, y - 1);
          const float* JXL_RESTRICT ry = dc.ConstPlaneRow(1, y);
          const float* JXL_RESTRICT ry_p = dc.ConstPlaneRow(1, y - 1);
          for (size_t x = 1; x < xsize; ++x) {
            const float vc = rc[x] - std::round(ry[x] * inv_fy) * yk;
            const float vn = rc_p[x] - std::round(ry_p[x] * inv_fy) * yk;
            const float vw = rc[x - 1] - std::round(ry[x - 1] * inv_fy) * yk;
            const float vnw = rc_p[x - 1] - std::round(ry_p[x - 1] * inv_fy) * yk;
            const float abs_s =
                std::abs(vc - clamped_grad(vn, vw, vnw)) * inv_f;
            float q_abs = 0.0f;
            float bits = 0.0f;
            if (abs_s >= dz) {
              q_abs = std::round(abs_s - bias1);
              if (q_abs > 2.0f) {
                q_abs = std::round(abs_s * 0.5f - bias2) * 2.0f;
              }
              if (q_abs > 0.0f) {
                bits = 1.25f + std::log2(1.0f + std::min(32.0f, q_abs));
              }
            }
            const float err = abs_s - q_abs;
            cost += bits + lambda_dist * err * err;
          }
        }
        return cost;
      };

      const double cost_x0 = eval_cfl_cost(0, inv_dc_x, 0);
      const double cost_b0 = eval_cfl_cost(2, inv_dc_b, 0);
      const double min_ch_gain =
          12.0 + 0.004 * static_cast<double>((xsize - 1) * (ysize - 1));
      int32_t best_x = 0;
      double best_cost_x = cost_x0;
      const int32_t cands_x[] = {est_x / 2, (est_x * 3) / 4, est_x - 1, est_x,
                                 est_x + 1};
      for (int32_t raw_cand : cands_x) {
        const int32_t cand = Clamp1<int32_t>(raw_cand, -128, 127);
        if (cand == 0 || cand == best_x) continue;
        const double c = eval_cfl_cost(0, inv_dc_x, cand);
        if (c < best_cost_x) {
          best_cost_x = c;
          best_x = cand;
        }
      }
      if (cost_x0 - best_cost_x < min_ch_gain) {
        best_x = 0;
        best_cost_x = cost_x0;
      }
      int32_t best_b = 0;
      double best_cost_b = cost_b0;
      const int32_t cands_b[] = {est_b / 2, (est_b * 3) / 4, est_b - 1, est_b,
                                 est_b + 1};
      for (int32_t raw_cand : cands_b) {
        const int32_t cand = Clamp1<int32_t>(raw_cand, -128, 127);
        if (cand == 0 || cand == best_b) continue;
        const double c = eval_cfl_cost(2, inv_dc_b, cand);
        if (c < best_cost_b) {
          best_cost_b = c;
          best_b = cand;
        }
      }
      if (cost_b0 - best_cost_b < min_ch_gain) {
        best_b = 0;
        best_cost_b = cost_b0;
      }
      const double min_tot_gain =
          55.0 + (enc_state->streaming_mode ? 0.015 : 0.004) *
                     static_cast<double>((xsize - 1) * (ysize - 1));
      if ((cost_x0 - best_cost_x) + (cost_b0 - best_cost_b) > min_tot_gain) {
        shared.cmap.base().SetYToXDC(best_x);
        shared.cmap.base().SetYToBDC(best_b);
      }
    }
    auto compute_dc_coeffs = [&](int group_index, int /* thread */) -> Status {
      const Rect r = enc_state->shared.frame_dim.DCGroupRect(group_index);
      int modular_group_index = group_index;
      if (enc_state->streaming_mode) {
        JXL_ENSURE(group_index == 0);
        modular_group_index = enc_state->dc_group_index;
      }
      JXL_RETURN_IF_ERROR(modular_frame_encoder->AddVarDCTDC(
          frame_header, dc, r, modular_group_index,
          enc_state->cparams.speed_tier < SpeedTier::kFalcon, enc_state,
          /*jpeg_transcode=*/false));
      return true;
    };
    JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, shared.frame_dim.num_dc_groups,
                                  ThreadPool::NoInit, compute_dc_coeffs,
                                  "Compute DC coeffs"));
    // TODO(veluca): this is only useful in tests and if inspection is enabled.
    if (!(frame_header.flags & FrameHeader::kSkipAdaptiveDCSmoothing)) {
      JXL_RETURN_IF_ERROR(AdaptiveDCSmoothing(
          memory_manager, shared.quantizer.MulDC(), &shared.dc_storage, pool));
    }
  }
  return true;
}

}  // namespace jxl
