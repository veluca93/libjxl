// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_chroma_from_luma.h"

#include <jxl/memory_manager.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <hwy/base.h>  // HWY_ALIGN_MAX
#include <limits>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/image.h"
#include "lib/jxl/quant_weights.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/enc_chroma_from_luma.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/cms/opsin_params.h"
#include "lib/jxl/dec_transforms-inl.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/enc_transforms-inl.h"
#include "lib/jxl/quantizer.h"
#include "lib/jxl/simd_util.h"
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Abs;
using hwy::HWY_NAMESPACE::Ge;
using hwy::HWY_NAMESPACE::GetLane;
using hwy::HWY_NAMESPACE::IfThenElse;
using hwy::HWY_NAMESPACE::Lt;

static HWY_FULL(float) df;

struct CFLFunction {
  static constexpr float kCoeff = 1.f / 3;
  static constexpr float kThres = 100.0f;
  static constexpr float kInvColorFactor = 1.0f / kDefaultColorFactor;
  CFLFunction(const float* values_m, const float* values_s, size_t num,
              float base, float distance_mul, float x_pred = 0.0f,
              float zero_lambda = 0.0f, float pred_lambda = 0.0f)
      : values_m(values_m),
        values_s(values_s),
        num(num),
        base(base),
        distance_mul(distance_mul),
        x_pred(x_pred),
        zero_lambda(zero_lambda),
        pred_lambda(pred_lambda) {
    JXL_DASSERT(num % Lanes(df) == 0);
  }

  // Returns f'(x), where f is 1/3 * sum ((|color residual| + 1)^2-1) +
  // distance_mul * x^2 * num.
  float Compute(float x, float eps, float* fpeps, float* fmeps) const {
    const float reg_hess = 2 * distance_mul * num + zero_lambda + pred_lambda;
    const float first_derivative =
        (2 * distance_mul * num + zero_lambda) * x + pred_lambda * (x - x_pred);
    const float first_derivative_peps = first_derivative + reg_hess * eps;
    const float first_derivative_meps = first_derivative - reg_hess * eps;

    const auto inv_color_factor = Set(df, kInvColorFactor);
    const auto thres = Set(df, kThres);
    const auto coeffx2 = Set(df, kCoeff * 2.0f);
    const auto one = Set(df, 1.0f);
    const auto zero = Set(df, 0.0f);
    const auto base_v = Set(df, base);
    const auto x_v = Set(df, x);
    const auto xpe_v = Set(df, x + eps);
    const auto xme_v = Set(df, x - eps);
    auto fd_v = Zero(df);
    auto fdpe_v = Zero(df);
    auto fdme_v = Zero(df);

    for (size_t i = 0; i < num; i += Lanes(df)) {
      // color residual = ax + b
      const auto a = Mul(inv_color_factor, Load(df, values_m + i));
      const auto b =
          Sub(Mul(base_v, Load(df, values_m + i)), Load(df, values_s + i));
      const auto v = MulAdd(a, x_v, b);
      const auto vpe = MulAdd(a, xpe_v, b);
      const auto vme = MulAdd(a, xme_v, b);
      const auto av = Abs(v);
      const auto avpe = Abs(vpe);
      const auto avme = Abs(vme);
      const auto acoeffx2 = Mul(coeffx2, a);
      auto d = Mul(acoeffx2, Add(av, one));
      auto dpe = Mul(acoeffx2, Add(avpe, one));
      auto dme = Mul(acoeffx2, Add(avme, one));
      d = IfThenElse(Lt(v, zero), Sub(zero, d), d);
      dpe = IfThenElse(Lt(vpe, zero), Sub(zero, dpe), dpe);
      dme = IfThenElse(Lt(vme, zero), Sub(zero, dme), dme);
      const auto above = Ge(av, thres);
      // TODO(eustas): use IfThenElseZero
      fd_v = Add(fd_v, IfThenElse(above, zero, d));
      fdpe_v = Add(fdpe_v, IfThenElse(above, zero, dpe));
      fdme_v = Add(fdme_v, IfThenElse(above, zero, dme));
    }

    *fpeps = first_derivative_peps + GetLane(SumOfLanes(df, fdpe_v));
    *fmeps = first_derivative_meps + GetLane(SumOfLanes(df, fdme_v));
    return first_derivative + GetLane(SumOfLanes(df, fd_v));
  }

  const float* JXL_RESTRICT values_m;
  const float* JXL_RESTRICT values_s;
  size_t num;
  float base;
  float distance_mul;
  float x_pred;
  float zero_lambda;
  float pred_lambda;
};

// Chroma-from-luma search, values_m will have luma -- and values_s chroma.
int32_t FindBestMultiplier(const float* values_m, const float* values_s,
                           size_t num, float base, float distance_mul,
                           bool fast, bool shrink_towards_zero = false,
                           float x_pred = 0.0f, float zero_lambda = 0.0f,
                           float pred_lambda = 0.0f) {
  if (num == 0) {
    return 0;
  }
  float x;
  float last_ddf = 0.0f;
  if (fast) {
    static constexpr float kInvColorFactor = 1.0f / kDefaultColorFactor;
    auto ca = Zero(df);
    auto cb = Zero(df);
    const auto inv_color_factor = Set(df, kInvColorFactor);
    const auto base_v = Set(df, base);
    for (size_t i = 0; i < num; i += Lanes(df)) {
      // color residual = ax + b
      const auto a = Mul(inv_color_factor, Load(df, values_m + i));
      const auto b =
          Sub(Mul(base_v, Load(df, values_m + i)), Load(df, values_s + i));
      ca = MulAdd(a, a, ca);
      cb = MulAdd(a, b, cb);
    }
    // + distance_mul * x^2 * num
    x = -GetLane(SumOfLanes(df, cb)) /
        (GetLane(SumOfLanes(df, ca)) + num * distance_mul * 0.5f);
  } else {
    constexpr float eps = 100;
    constexpr float kClamp = 20.0f;
    CFLFunction fn(values_m, values_s, num, base, distance_mul, x_pred,
                   zero_lambda, pred_lambda);
    x = 0;
    // Up to 20 Newton iterations, with approximate derivatives.
    // Derivatives are approximate due to the high amount of noise in the exact
    // derivatives.
    for (size_t i = 0; i < 20; i++) {
      float dfpeps;
      float dfmeps;
      float d_f = fn.Compute(x, eps, &dfpeps, &dfmeps);
      float ddf = (dfpeps - dfmeps) / (2 * eps);
      last_ddf = ddf;
      float kExperimentalInsignificantStabilizer = 0.85;
      float step = d_f / (ddf + kExperimentalInsignificantStabilizer);
      x -= std::min(kClamp, std::max(-kClamp, step));
      if (std::abs(step) < 3e-3) break;
    }
  }
  if (shrink_towards_zero) {
    constexpr float kTowardsZero = 2.6f;
    if (std::abs(x) <= kTowardsZero) {
      x = 0.0f;
    } else if (x > 0.0f) {
      x -= kTowardsZero;
    } else {
      x += kTowardsZero;
    }
  } else if (pred_lambda > 0.0f) {
    const float diff_pred = x - x_pred;
    if (std::abs(diff_pred) <= 1.05f &&
        last_ddf * diff_pred * diff_pred < 950.0f) {
      x = x_pred;
    }
  }
  return jxl::Clamp1(std::round(x), -128.0f, 127.0f);
}

Status ComputeTile(const Image3F& opsin, const Rect& opsin_rect,
                   const DequantMatrices& dequant,
                   const AcStrategyImage* ac_strategy,
                   const ImageI* raw_quant_field, const Quantizer* quantizer,
                   const Rect& rect, bool fast, bool use_dct8,
                   bool shrink_towards_zero, float x_qm_multiplier,
                   float b_qm_multiplier, ImageSB* map_x, ImageSB* map_b,
                   Span<float> mem) {
  static_assert(kEncTileDimInBlocks == kColorTileDimInBlocks,
                "Invalid color tile dim");
  constexpr float kDistanceMultiplierAC = 1e-9f;
  const size_t dct_scratch_size =
      3 * (MaxVectorSize() / sizeof(float)) * AcStrategy::kMaxBlockDim;

  const size_t y0 = rect.y0();
  const size_t x0 = rect.x0();
  const size_t x1 = rect.x0() + rect.xsize();
  const size_t y1 = rect.y0() + rect.ysize();

  int ty = y0 / kColorTileDimInBlocks;
  int tx = x0 / kColorTileDimInBlocks;

  int8_t* JXL_RESTRICT row_out_x = map_x->Row(ty);
  int8_t* JXL_RESTRICT row_out_b = map_b->Row(ty);

  // All are aligned.
  float* HWY_RESTRICT block_y = mem.begin();
  float* HWY_RESTRICT block_x = block_y + AcStrategy::kMaxCoeffArea;
  float* HWY_RESTRICT block_b = block_x + AcStrategy::kMaxCoeffArea;
  JXL_ENSURE(mem.remove_prefix(3 * AcStrategy::kMaxCoeffArea));
  constexpr size_t kMaxTileCoeffs = 4 * kColorTileDim * kColorTileDim;
  float* HWY_RESTRICT coeffs_yx = mem.begin();
  float* HWY_RESTRICT coeffs_x = coeffs_yx + kMaxTileCoeffs;
  float* HWY_RESTRICT coeffs_yb = coeffs_x + kMaxTileCoeffs;
  float* HWY_RESTRICT coeffs_b = coeffs_yb + kMaxTileCoeffs;
  JXL_ENSURE(mem.remove_prefix(4 * kMaxTileCoeffs));
  float* HWY_RESTRICT scratch_space = mem.begin();
  JXL_ENSURE(mem.size() == 2 * AcStrategy::kMaxCoeffArea + dct_scratch_size);

  size_t num_ac = 0;

  for (size_t y = y0; y < y1; ++y) {
    const float* JXL_RESTRICT row_y =
        opsin_rect.ConstPlaneRow(opsin, 1, y * kBlockDim);
    const float* JXL_RESTRICT row_x =
        opsin_rect.ConstPlaneRow(opsin, 0, y * kBlockDim);
    const float* JXL_RESTRICT row_b =
        opsin_rect.ConstPlaneRow(opsin, 2, y * kBlockDim);
    size_t stride = opsin.PixelsPerRow();

    for (size_t x = x0; x < x1; x++) {
      AcStrategy acs = use_dct8
                           ? AcStrategy::FromRawStrategy(AcStrategyType::DCT)
                           : ac_strategy->ConstRow(y)[x];
      if (!acs.IsFirstBlock()) continue;
      TransformFromPixels(acs.Strategy(), row_y + x * kBlockDim, stride,
                          block_y, scratch_space);
      TransformFromPixels(acs.Strategy(), row_x + x * kBlockDim, stride,
                          block_x, scratch_space);
      TransformFromPixels(acs.Strategy(), row_b + x * kBlockDim, stride,
                          block_b, scratch_space);
      const float* const JXL_RESTRICT qm_x =
          dequant.InvMatrix(acs.Strategy(), 0);
      const float* const JXL_RESTRICT qm_b =
          dequant.InvMatrix(acs.Strategy(), 2);

      // Copy AC coefficients in the local block. The order in which
      // coefficients get stored does not matter.
      size_t cx = acs.covered_blocks_x();
      size_t cy = acs.covered_blocks_y();
      CoefficientLayout(&cy, &cx);
      // Zero out LFs. This introduces terms in the optimization loop that
      // don't affect the result, as they are all 0, but allow for simpler
      // SIMDfication.
      for (size_t iy = 0; iy < cy; iy++) {
        for (size_t ix = 0; ix < cx; ix++) {
          block_y[cx * kBlockDim * iy + ix] = 0;
          block_x[cx * kBlockDim * iy + ix] = 0;
          block_b[cx * kBlockDim * iy + ix] = 0;
        }
      }
      // Unclear why this is like it is. (This works slightly better
      // than the previous approach which was also a hack.)
      const float qq =
          (raw_quant_field == nullptr) ? 1.0f : raw_quant_field->ConstRow(y)[x];
      // Experimentally values 128-130 seem best -- I don't know why we
      // need this multiplier.
      const float kStrangeMultiplier = 128;
      float q = (quantizer == nullptr)
                    ? 1.0f
                    : quantizer->Scale() * kStrangeMultiplier * qq;
      const auto qv_x = Set(df, q * x_qm_multiplier);
      const auto qv_b = Set(df, q * b_qm_multiplier);
      for (size_t i = 0; i < cx * cy * 64; i += Lanes(df)) {
        const auto b_y = Load(df, block_y + i);
        const auto b_x = Load(df, block_x + i);
        const auto b_b = Load(df, block_b + i);
        const auto qqm_x = Mul(qv_x, Load(df, qm_x + i));
        const auto qqm_b = Mul(qv_b, Load(df, qm_b + i));
        Store(Mul(b_y, qqm_x), df, coeffs_yx + num_ac);
        Store(Mul(b_x, qqm_x), df, coeffs_x + num_ac);
        Store(Mul(b_y, qqm_b), df, coeffs_yb + num_ac);
        Store(Mul(b_b, qqm_b), df, coeffs_b + num_ac);
        num_ac += Lanes(df);
      }
    }
  }
  JXL_ENSURE(num_ac % Lanes(df) == 0);
  const bool use_reg = (!fast && !shrink_towards_zero);
  const bool use_pred = (use_reg && !use_dct8);
  auto compute_pred = [&](const ImageSB* map, float* out_pred_lambda) -> float {
    if (!use_pred) {
      *out_pred_lambda = 0.0f;
      return 0.0f;
    }
    const int lx = tx & 3;
    const int ly = ty & 3;
    if (lx == 0 && ly == 0) {
      *out_pred_lambda = 3.0f;
      return 0.0f;
    }
    const int8_t* row_cur = map->ConstRow(ty);
    const int8_t* row_top = (ly > 0) ? map->ConstRow(ty - 1) : nullptr;
    const int32_t left =
        (lx > 0) ? row_cur[tx - 1] : ((ly > 0) ? row_top[tx] : 0);
    const int32_t top = (ly > 0) ? row_top[tx] : left;
    const int32_t topleft = (lx > 0 && ly > 0) ? row_top[tx - 1] : left;
    const int32_t m = std::min(top, left);
    const int32_t M = std::max(top, left);
    const int32_t grad = top + left - topleft;
    const int32_t spread =
        std::max(M, topleft) - std::min(m, topleft);
    *out_pred_lambda = 10.0f / (1.0f + 0.15f * static_cast<float>(spread));
    return static_cast<float>(std::min(M, std::max(m, grad)));
  };
  const float zero_lambda = use_reg ? 2.5f : 0.0f;
  float pred_lambda_x = 0.0f;
  float pred_lambda_b = 0.0f;
  const float pred_x = compute_pred(map_x, &pred_lambda_x);
  const float pred_b = compute_pred(map_b, &pred_lambda_b);
  row_out_x[tx] = FindBestMultiplier(
      coeffs_yx, coeffs_x, num_ac, 0.0f, kDistanceMultiplierAC, fast,
      shrink_towards_zero, pred_x, zero_lambda, pred_lambda_x);
  row_out_b[tx] = FindBestMultiplier(
      coeffs_yb, coeffs_b, num_ac, jxl::cms::kYToBRatio, kDistanceMultiplierAC,
      fast, shrink_towards_zero, pred_b, zero_lambda, pred_lambda_b);
  return true;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(ComputeTile);

Status CfLHeuristics::ComputeTile(const Rect& r, const Image3F& opsin,
                                  const Rect& opsin_rect,
                                  const DequantMatrices& dequant,
                                  const AcStrategyImage* ac_strategy,
                                  const ImageI* raw_quant_field,
                                  const Quantizer* quantizer, bool fast,
                                  size_t thread, ColorCorrelationMap* cmap,
                                  bool shrink_towards_zero,
                                  float x_qm_multiplier,
                                  float b_qm_multiplier) {
  bool use_dct8 = ac_strategy == nullptr;
  Span<float> scratch(mem.address<float>() + thread * ItemsPerThread(),
                      ItemsPerThread());
  return HWY_DYNAMIC_DISPATCH(ComputeTile)(
      opsin, opsin_rect, dequant, ac_strategy, raw_quant_field, quantizer, r,
      fast, use_dct8, shrink_towards_zero, x_qm_multiplier, b_qm_multiplier,
      &cmap->ytox_map, &cmap->ytob_map, scratch);
}

Status ColorCorrelationEncodeDC(const ColorCorrelation& color_correlation,
                                BitWriter* writer, LayerType layer,
                                AuxOut* aux_out) {
  float color_factor = color_correlation.GetColorFactor();
  float base_correlation_x = color_correlation.GetBaseCorrelationX();
  float base_correlation_b = color_correlation.GetBaseCorrelationB();
  int32_t ytox_dc = color_correlation.GetYToXDC();
  int32_t ytob_dc = color_correlation.GetYToBDC();

  return writer->WithMaxBits(
      1 + 2 * kBitsPerByte + 12 + 32, layer, aux_out, [&]() -> Status {
        if (ytox_dc == 0 && ytob_dc == 0 &&
            color_factor == kDefaultColorFactor && base_correlation_x == 0.0f &&
            base_correlation_b == jxl::cms::kYToBRatio) {
          writer->Write(1, 1);
          return true;
        }
        writer->Write(1, 0);
        JXL_RETURN_IF_ERROR(
            U32Coder::Write(kColorFactorDist, color_factor, writer));
        JXL_RETURN_IF_ERROR(F16Coder::Write(base_correlation_x, writer));
        JXL_RETURN_IF_ERROR(F16Coder::Write(base_correlation_b, writer));
        writer->Write(kBitsPerByte,
                      ytox_dc - std::numeric_limits<int8_t>::min());
        writer->Write(kBitsPerByte,
                      ytob_dc - std::numeric_limits<int8_t>::min());
        return true;
      });
}

}  // namespace jxl
#endif  // HWY_ONCE
