// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_chroma_from_luma.h"

#include <jxl/memory_manager.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <hwy/base.h>  // HWY_ALIGN_MAX
#include <limits>
#include <vector>

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
#include "lib/jxl/base/fast_math-inl.h"
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
using hwy::HWY_NAMESPACE::CopySignToAbs;
using hwy::HWY_NAMESPACE::Div;
using hwy::HWY_NAMESPACE::Floor;
using hwy::HWY_NAMESPACE::Ge;
using hwy::HWY_NAMESPACE::GetLane;
using hwy::HWY_NAMESPACE::IfThenElse;
using hwy::HWY_NAMESPACE::IfThenElseZero;
using hwy::HWY_NAMESPACE::Lt;
using hwy::HWY_NAMESPACE::Max;
using hwy::HWY_NAMESPACE::Min;
using hwy::HWY_NAMESPACE::Mul;
using hwy::HWY_NAMESPACE::MulAdd;
using hwy::HWY_NAMESPACE::NegMulAdd;
using hwy::HWY_NAMESPACE::Sub;

static HWY_FULL(float) df;

struct CFLFunction {
  static constexpr float kCoeff = 1.f / 3;
  static constexpr float kThres = 100.0f;
  CFLFunction(const float* values_m, const float* values_s, size_t num,
              float inv_color_factor, float base, float distance_mul,
              float x_pred = 0.0f, float zero_lambda = 0.0f,
              float pred_lambda = 0.0f)
      : values_m(values_m),
        values_s(values_s),
        num(num),
        inv_color_factor(inv_color_factor),
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

    const auto inv_color_factor_v = Set(df, inv_color_factor);
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
      const auto a = Mul(inv_color_factor_v, Load(df, values_m + i));
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
  float inv_color_factor;
  float base;
  float distance_mul;
  float x_pred;
  float zero_lambda;
  float pred_lambda;
};

// Chroma-from-luma search, values_m will have luma -- and values_s chroma.
int32_t FindBestMultiplier(const float* values_m, const float* values_s,
                           size_t num, float inv_color_factor, float base,
                           float distance_mul, bool fast,
                           bool shrink_towards_zero = false,
                           float x_pred = 0.0f, float zero_lambda = 0.0f,
                           float pred_lambda = 0.0f) {
  if (num == 0) {
    return 0;
  }
  float x;
  float last_ddf = 0.0f;
  if (fast) {
    auto ca = Zero(df);
    auto cb = Zero(df);
    const auto inv_color_factor_v = Set(df, inv_color_factor);
    const auto base_v = Set(df, base);
    for (size_t i = 0; i < num; i += Lanes(df)) {
      // color residual = ax + b
      const auto a = Mul(inv_color_factor_v, Load(df, values_m + i));
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
    CFLFunction fn(values_m, values_s, num, inv_color_factor, base,
                   distance_mul, x_pred, zero_lambda, pred_lambda);
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

constexpr float kCflRdLambda = 0.08086717f * 0.25f * 1.375f;  // 0.02779809f
constexpr float kCflBaseWeightX = 0.30f;
constexpr float kCflBaseWeightB = 0.28f;
constexpr float kCflNonZeroBits = 4.0f;
constexpr float kCflMagnitudeBits = 1.5f;
constexpr float kRecYBlendDct8 = 0.65f;
constexpr float kRecYBlendFinal = 1.0f;

static inline float ChromaRdoWeight(float distance) {
  return 2.0f - jxl::Clamp1((distance - 2.25f) * (1.0f / 1.75f), 0.0f, 1.0f);
}

static inline float CmapResidualBits(int32_t res) {
  const int32_t abs_res = std::abs(res);
  if (abs_res == 0) return 0.8f;
  if (abs_res == 1) return 2.0f;
  return 2.5f + std::log2(static_cast<float>(abs_res));
}

static inline int32_t ComputeTilePred(const ImageSB* map, int tx, int ty) {
  const int lx = tx & 3;
  const int ly = ty & 3;
  if (lx == 0 && ly == 0) return 0;
  const int8_t* row_cur = map->ConstRow(ty);
  const int8_t* row_top = (ly > 0) ? map->ConstRow(ty - 1) : nullptr;
  if (lx == 0) return row_top[tx];
  if (ly == 0) return row_cur[tx - 1];
  const int32_t left = row_cur[tx - 1];
  const int32_t top = row_top[tx];
  const int32_t topleft = row_top[tx - 1];
  const int32_t m = std::min(top, left);
  const int32_t M = std::max(top, left);
  return std::min(M, std::max(m, top + left - topleft));
}

float ComputeCflRdCost(const float* JXL_RESTRICT values_m,
                       const float* JXL_RESTRICT values_s, size_t num,
                       float factor, float weight, float lambda,
                       float bias_one) {
  JXL_DASSERT(num % Lanes(df) == 0);
  const auto factor_v = Set(df, factor);
  const auto weight_v = Set(df, weight);
  const auto lambda_v = Set(df, lambda);
  const auto bias_one_v = Set(df, bias_one);
  const auto bias_num_v = Set(df, kDefaultQuantBias[3]);
  const auto nz_bits_v = Set(df, kCflNonZeroBits);
  const auto mag_bits_v = Set(df, kCflMagnitudeBits);
  const auto half_v = Set(df, 0.5f);
  const auto one_v = Set(df, 1.0f);
  const auto one_p5_v = Set(df, 1.5f);
  auto total_cost_v = Zero(df);

  for (size_t i = 0; i < num; i += Lanes(df)) {
    const auto m = Load(df, values_m + i);
    const auto s = Load(df, values_s + i);
    const auto res = NegMulAdd(factor_v, m, s);
    const auto abs_res = Abs(res);
    const auto zero_cost = Mul(weight_v, Mul(abs_res, abs_res));

    const auto abs_level = Floor(Add(abs_res, half_v));
    const auto abs_level_safe = Max(abs_level, one_v);
    const auto is_one = Lt(abs_level, one_p5_v);
    const auto abs_deq =
        IfThenElse(is_one, bias_one_v,
                   Sub(abs_level_safe, Div(bias_num_v, abs_level_safe)));
    const auto coded_err = Sub(abs_res, abs_deq);
    const auto bits =
        MulAdd(mag_bits_v, FastLog2f(df, abs_level_safe), nz_bits_v);
    const auto coded_cost =
        MulAdd(weight_v, Mul(coded_err, coded_err), Mul(lambda_v, bits));
    const auto best_cost =
        IfThenElse(Ge(abs_level, one_v), Min(zero_cost, coded_cost), zero_cost);
    total_cost_v = Add(total_cost_v, best_cost);
  }
  return GetLane(SumOfLanes(df, total_cost_v));
}

int8_t FindBestMultiplierDiscrete(const float* JXL_RESTRICT values_m,
                                  const float* JXL_RESTRICT values_s,
                                  size_t num, float color_factor,
                                  int32_t pred_cand, bool use_pred,
                                  float weight, float lambda, float bias_one) {
  if (num == 0) return 0;
  auto sum_ms_v = Zero(df);
  auto sum_mm_v = Zero(df);
  for (size_t i = 0; i < num; i += Lanes(df)) {
    const auto m = Load(df, values_m + i);
    const auto s = Load(df, values_s + i);
    sum_ms_v = MulAdd(m, s, sum_ms_v);
    sum_mm_v = MulAdd(m, m, sum_mm_v);
  }
  const float dot_ms = GetLane(SumOfLanes(df, sum_ms_v));
  const float dot_mm = GetLane(SumOfLanes(df, sum_mm_v));
  const float target_factor = (dot_mm > 1e-6f) ? (dot_ms / dot_mm) : 0.0f;
  const int32_t target_cand = static_cast<int32_t>(
      jxl::Clamp1(std::round(target_factor * color_factor), -128.0f, 127.0f));

  int32_t candidates[31];
  size_t num_candidates = 0;
  auto push_cand = [&](int32_t c) {
    const int32_t clamped = jxl::Clamp1(c, -128, 127);
    for (size_t i = 0; i < num_candidates; ++i) {
      if (candidates[i] == clamped) return;
    }
    candidates[num_candidates++] = clamped;
  };

  push_cand(0);
  if (use_pred) push_cand(pred_cand);
  push_cand(target_cand);
  for (int32_t d = 1; d <= 4; ++d) {
    push_cand(target_cand - d);
    push_cand(target_cand + d);
    if (use_pred) {
      push_cand(pred_cand - d);
      push_cand(pred_cand + d);
    }
  }
  if (color_factor > 100.0f) {
    for (int32_t d : {6, 8, 11}) {
      push_cand(target_cand - d);
      push_cand(target_cand + d);
      if (use_pred) {
        push_cand(pred_cand - d);
        push_cand(pred_cand + d);
      }
    }
  }

  const float inv_color_factor = 1.0f / color_factor;
  int32_t best_cand = use_pred ? pred_cand : 0;
  float best_cost = std::numeric_limits<float>::infinity();
  for (size_t i = 0; i < num_candidates; ++i) {
    const int32_t cand = candidates[i];
    const float rd_cost = ComputeCflRdCost(
        values_m, values_s, num, static_cast<float>(cand) * inv_color_factor,
        weight, lambda, bias_one);
    const float map_bits =
        use_pred ? CmapResidualBits(cand - pred_cand)
                 : 0.5f * CmapResidualBits(cand);
    const float total_cost = rd_cost + lambda * map_bits;
    if (total_cost < best_cost) {
      best_cost = total_cost;
      best_cand = cand;
    }
  }
  return static_cast<int8_t>(best_cand);
}

Status ComputeTile(const Image3F& opsin, const Rect& opsin_rect,
                   const DequantMatrices& dequant,
                   const AcStrategyImage* ac_strategy,
                   const ImageI* raw_quant_field, const Quantizer* quantizer,
                   const Rect& rect, bool fast, bool use_dct8,
                   bool shrink_towards_zero, float x_qm_multiplier,
                   float b_qm_multiplier, float butteraugli_distance,
                   float color_factor, float base_x, float base_b,
                   ImageSB* map_x, ImageSB* map_b, Span<float> mem) {
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

  const bool use_discrete_rdo =
      (!fast && !shrink_towards_zero && quantizer != nullptr &&
       raw_quant_field != nullptr);

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
      const float qq =
          (raw_quant_field == nullptr) ? 1.0f : raw_quant_field->ConstRow(y)[x];
      if (use_discrete_rdo) {
        const float* const JXL_RESTRICT qm_y =
            dequant.InvMatrix(acs.Strategy(), 1);
        const float* const JXL_RESTRICT dm_y =
            dequant.Matrix(acs.Strategy(), 1);
        float* HWY_RESTRICT t0_buf = scratch_space;
        const size_t row_stride = cx * kBlockDim;
        for (size_t iy = 0; iy < cy * kBlockDim; ++iy) {
          const bool low_v = (iy < cy * 4);
          float* HWY_RESTRICT row = t0_buf + iy * row_stride;
          for (size_t ix = 0; ix < row_stride; ++ix) {
            row[ix] = (low_v && ix < cx * 4) ? 0.58f : 0.64f;
          }
        }
        const float q_y = quantizer->Scale() * qq;
        const float inv_q_y = 1.0f / q_y;
        const auto qv_x = Set(df, q_y * x_qm_multiplier);
        const auto qv_b = Set(df, q_y * b_qm_multiplier);
        const float rec_blend = use_dct8 ? kRecYBlendDct8 : kRecYBlendFinal;
        for (size_t i = 0; i < cx * cy * 64; i += Lanes(df)) {
          const auto b_y_raw = Load(df, block_y + i);
          const auto b_x_raw = Load(df, block_x + i);
          const auto b_b_raw = Load(df, block_b + i);

          const auto qm_y_v = Load(df, qm_y + i);
          const auto dm_y_v = Load(df, dm_y + i);
          const auto val_y = Mul(b_y_raw, Mul(qm_y_v, Set(df, q_y)));
          const auto abs_val_y = Abs(val_y);
          const auto t0_y_v = Load(df, t0_buf + i);
          const auto abs_q_y = IfThenElseZero(
              Ge(abs_val_y, t0_y_v), Floor(Add(abs_val_y, Set(df, 0.5f))));
          const auto abs_q_y_safe = Max(abs_q_y, Set(df, 1.0f));
          const auto abs_adj_y = IfThenElse(
              Lt(abs_q_y, Set(df, 1.5f)),
              IfThenElseZero(Ge(abs_q_y, Set(df, 1.0f)),
                             Set(df, kDefaultQuantBias[1])),
              Sub(abs_q_y_safe,
                  Div(Set(df, kDefaultQuantBias[3]), abs_q_y_safe)));
          const auto adj_y = CopySignToAbs(abs_adj_y, val_y);
          const auto deq_y = Mul(adj_y, Mul(dm_y_v, Set(df, inv_q_y)));
          const auto b_y_rec = MulAdd(Set(df, rec_blend), deq_y,
                                      Mul(Set(df, 1.0f - rec_blend), b_y_raw));
          const auto b_x_opp = NegMulAdd(Set(df, base_x), b_y_rec, b_x_raw);
          const auto b_b_opp = NegMulAdd(Set(df, base_b), b_y_rec, b_b_raw);

          const auto qqm_x = Mul(qv_x, Load(df, qm_x + i));
          const auto qqm_b = Mul(qv_b, Load(df, qm_b + i));
          Store(Mul(b_y_rec, qqm_x), df, coeffs_yx + num_ac);
          Store(Mul(b_x_opp, qqm_x), df, coeffs_x + num_ac);
          Store(Mul(b_y_rec, qqm_b), df, coeffs_yb + num_ac);
          Store(Mul(b_b_opp, qqm_b), df, coeffs_b + num_ac);
          num_ac += Lanes(df);
        }
      } else {
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
  }
  JXL_ENSURE(num_ac % Lanes(df) == 0);
  if (use_discrete_rdo) {
    const bool use_pred = !use_dct8;
    const int32_t pred_x = use_pred ? ComputeTilePred(map_x, tx, ty) : 0;
    const int32_t pred_b = use_pred ? ComputeTilePred(map_b, tx, ty) : 0;
    const float chroma_w = ChromaRdoWeight(butteraugli_distance);
    row_out_x[tx] = FindBestMultiplierDiscrete(
        coeffs_yx, coeffs_x, num_ac, color_factor, pred_x, use_pred,
        kCflBaseWeightX * chroma_w, kCflRdLambda, kDefaultQuantBias[0]);
    row_out_b[tx] = FindBestMultiplierDiscrete(
        coeffs_yb, coeffs_b, num_ac, color_factor, pred_b, use_pred,
        kCflBaseWeightB * chroma_w, kCflRdLambda, kDefaultQuantBias[2]);
    return true;
  }
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
  const float inv_color_factor = 1.0f / color_factor;
  row_out_x[tx] = FindBestMultiplier(
      coeffs_yx, coeffs_x, num_ac, inv_color_factor, base_x,
      kDistanceMultiplierAC, fast, shrink_towards_zero, pred_x, zero_lambda,
      pred_lambda_x);
  row_out_b[tx] = FindBestMultiplier(
      coeffs_yb, coeffs_b, num_ac, inv_color_factor, base_b,
      kDistanceMultiplierAC, fast, shrink_towards_zero, pred_b, zero_lambda,
      pred_lambda_b);
  return true;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

namespace {

float DecodeF16BitsExact(uint32_t bits16) {
  const uint32_t sign = bits16 >> 15;
  const uint32_t biased_exp = (bits16 >> 10) & 0x1F;
  const uint32_t mantissa = bits16 & 0x3FF;
  if (biased_exp == 31) return 0.0f;
  if (biased_exp == 0) {
    float v = (1.0f / 16384.0f) * (static_cast<float>(mantissa) * (1.0f / 1024.0f));
    return sign ? -v : v;
  }
  const uint32_t biased_exp32 = biased_exp + (127 - 15);
  const uint32_t mantissa32 = mantissa << (23 - 10);
  const uint32_t bits32 = (sign << 31) | (biased_exp32 << 23) | mantissa32;
  float out = 0.0f;
  memcpy(&out, &bits32, sizeof(out));
  return out;
}

uint32_t EncodeF16BitsTrunc(float value) {
  uint32_t bits32 = 0;
  memcpy(&bits32, &value, sizeof(bits32));
  const uint32_t sign = bits32 >> 31;
  const uint32_t biased_exp32 = (bits32 >> 23) & 0xFF;
  const uint32_t mantissa32 = bits32 & 0x7FFFFF;
  const int32_t exp = static_cast<int32_t>(biased_exp32) - 127;
  if (exp > 15) return (sign << 15) | (30u << 10) | 0x3FFu;
  if (exp < -24) return 0u;
  uint32_t biased_exp16 = 0;
  uint32_t mantissa16 = 0;
  if (exp < -14) {
    const uint32_t sub_exp = static_cast<uint32_t>(-14 - exp);
    mantissa16 = (1u << (10 - sub_exp)) + (mantissa32 >> (13 + sub_exp));
  } else {
    biased_exp16 = static_cast<uint32_t>(exp + 15);
    mantissa16 = mantissa32 >> 13;
  }
  return (sign << 15) | (biased_exp16 << 10) | (mantissa16 & 0x3FFu);
}

float RoundToF16Exact(float value) {
  value = jxl::Clamp1(value, -3.9f, 3.9f);
  const uint32_t b0 = EncodeF16BitsTrunc(value);
  const float f0 = DecodeF16BitsExact(b0);
  const uint32_t mag0 = b0 & 0x7FFFu;
  if (mag0 >= 0x7BFFu) return f0;
  const uint32_t b1 = (b0 & 0x8000u) | (mag0 + 1u);
  const float f1 = DecodeF16BitsExact(b1);
  if (EncodeF16BitsTrunc(f1) == b1 &&
      std::abs(f1 - value) < std::abs(f0 - value)) {
    return f1;
  }
  return f0;
}

double CmapResidualBitsScalar(int32_t res) {
  const int32_t abs_res = std::abs(res);
  if (abs_res == 0) return 0.8;
  if (abs_res == 1) return 2.0;
  return 2.5 + std::log2(static_cast<double>(abs_res));
}

}  // namespace

HWY_EXPORT(ComputeTile);

void OptimizeColorCorrelationParams(
    const Image3F& opsin, const Rect& opsin_rect,
    const FrameDimensions& frame_dim, const DequantMatrices& dequant,
    const Quantizer& quantizer, const ImageF& initial_quant_field,
    float x_qm_multiplier, float b_qm_multiplier, float butteraugli_distance,
    ColorCorrelationMap* cmap) {
  const size_t xblocks = frame_dim.xsize_blocks;
  const size_t yblocks = frame_dim.ysize_blocks;
  if (xblocks * yblocks <= 64) return;

  const size_t tx_num = cmap->ytox_map.xsize();
  const size_t ty_num = cmap->ytox_map.ysize();
  const size_t num_tiles = tx_num * ty_num;
  if (num_tiles == 0) return;

  const float* qm_x_dct = dequant.InvMatrix(AcStrategyType::DCT, 0);
  const float* qm_b_dct = dequant.InvMatrix(AcStrategyType::DCT, 2);
  float qm_x_sum = 0.0f;
  float qm_b_sum = 0.0f;
  size_t qm_cnt = 0;
  for (size_t v = 0; v < 8; ++v) {
    for (size_t u = 0; u < 8; ++u) {
      if (u + v >= 1 && u + v <= 3) {
        qm_x_sum += qm_x_dct[v * 8 + u];
        qm_b_sum += qm_b_dct[v * 8 + u];
        ++qm_cnt;
      }
    }
  }
  const float qm_x_avg = qm_x_sum / static_cast<float>(qm_cnt);
  const float qm_b_avg = qm_b_sum / static_cast<float>(qm_cnt);

  std::vector<double> tile_yy(num_tiles, 0.0);
  std::vector<double> tile_yx(num_tiles, 0.0);
  std::vector<double> tile_yb(num_tiles, 0.0);
  std::vector<double> tile_qy(num_tiles, 0.0);
  std::vector<size_t> tile_blks(num_tiles, 0);

  std::vector<float> dc_x(xblocks * yblocks, 0.0f);
  std::vector<float> dc_y(xblocks * yblocks, 0.0f);
  std::vector<float> dc_b(xblocks * yblocks, 0.0f);

  const size_t stride = opsin.PixelsPerRow();
  for (size_t by = 0; by < yblocks; ++by) {
    const float* JXL_RESTRICT row_x0 =
        opsin_rect.ConstPlaneRow(opsin, 0, by * kBlockDim);
    const float* JXL_RESTRICT row_y0 =
        opsin_rect.ConstPlaneRow(opsin, 1, by * kBlockDim);
    const float* JXL_RESTRICT row_b0 =
        opsin_rect.ConstPlaneRow(opsin, 2, by * kBlockDim);
    const float* JXL_RESTRICT qf_row = initial_quant_field.ConstRow(by);
    const size_t ty = by / kColorTileDimInBlocks;
    for (size_t bx = 0; bx < xblocks; ++bx) {
      double sx = 0.0;
      double sy = 0.0;
      double sb = 0.0;
      for (size_t iy = 0; iy < kBlockDim; ++iy) {
        const float* rx = row_x0 + iy * stride + bx * kBlockDim;
        const float* ry = row_y0 + iy * stride + bx * kBlockDim;
        const float* rb = row_b0 + iy * stride + bx * kBlockDim;
        for (size_t ix = 0; ix < kBlockDim; ++ix) {
          sx += rx[ix];
          sy += ry[ix];
          sb += rb[ix];
        }
      }
      const double mx = sx * (1.0 / 64.0);
      const double my = sy * (1.0 / 64.0);
      const double mb = sb * (1.0 / 64.0);
      dc_x[by * xblocks + bx] = static_cast<float>(mx);
      dc_y[by * xblocks + bx] = static_cast<float>(my);
      dc_b[by * xblocks + bx] = static_cast<float>(mb);

      double vyy = 0.0;
      double vyx = 0.0;
      double vyb = 0.0;
      for (size_t iy = 0; iy < kBlockDim; ++iy) {
        const float* rx = row_x0 + iy * stride + bx * kBlockDim;
        const float* ry = row_y0 + iy * stride + bx * kBlockDim;
        const float* rb = row_b0 + iy * stride + bx * kBlockDim;
        for (size_t ix = 0; ix < kBlockDim; ++ix) {
          const double dy = ry[ix] - my;
          vyy += dy * dy;
          vyx += dy * (rx[ix] - mx);
          vyb += dy * (rb[ix] - mb);
        }
      }
      const size_t tx = bx / kColorTileDimInBlocks;
      const size_t ti = ty * tx_num + tx;
      // Orthonormal 8x8 DCT Parseval factor is 1/64.
      tile_yy[ti] += vyy * (1.0 / 64.0);
      tile_yx[ti] += vyx * (1.0 / 64.0);
      tile_yb[ti] += vyb * (1.0 / 64.0);
      tile_qy[ti] += qf_row[bx];
      tile_blks[ti] += 1;
    }
  }

  // Estimate DC CfL correlation savings to know if ColorCorrelationEncodeDC's
  // 50-bit header will already be paid or unlocked by DC CfL.
  double dc_num_x = 0.0;
  double dc_den = 1e-9;
  double dc_num_b = 0.0;
  auto cgrad = [](float t, float l, float tl) -> float {
    const float mn = std::min(t, l);
    const float mx = std::max(t, l);
    return std::min(mx, std::max(mn, t + l - tl));
  };
  for (size_t by = 1; by < yblocks; ++by) {
    for (size_t bx = 1; bx < xblocks; ++bx) {
      const size_t i = by * xblocks + bx;
      const float dy =
          dc_y[i] - cgrad(dc_y[i - xblocks], dc_y[i - 1], dc_y[i - xblocks - 1]);
      if (std::abs(dy) < 1e-4f) continue;
      const float dx =
          dc_x[i] - cgrad(dc_x[i - xblocks], dc_x[i - 1], dc_x[i - xblocks - 1]);
      const float b0 = dc_b[i] - jxl::cms::kYToBRatio * dc_y[i];
      const float bt = dc_b[i - xblocks] - jxl::cms::kYToBRatio * dc_y[i - xblocks];
      const float bl = dc_b[i - 1] - jxl::cms::kYToBRatio * dc_y[i - 1];
      const float btl =
          dc_b[i - xblocks - 1] - jxl::cms::kYToBRatio * dc_y[i - xblocks - 1];
      const float db = b0 - cgrad(bt, bl, btl);
      dc_den += static_cast<double>(dy) * dy;
      dc_num_x += static_cast<double>(dy) * dx;
      dc_num_b += static_cast<double>(dy) * db;
    }
  }
  const double inv_dc_x = quantizer.GetInvDcStep(0);
  const double inv_dc_b = quantizer.GetInvDcStep(2);
  // Variance reduction in quantized DC residual steps -> approximate bits saved.
  const double dc_var_red_x =
      ((dc_num_x * dc_num_x) / dc_den) * (inv_dc_x * inv_dc_x);
  const double dc_var_red_b =
      ((dc_num_b * dc_num_b) / dc_den) * (inv_dc_b * inv_dc_b);
  const double est_dc_gain_bits =
      0.35 * std::min(dc_var_red_x, 4.0 * xblocks * yblocks) +
      0.35 * std::min(dc_var_red_b, 4.0 * xblocks * yblocks);
  const double eff_header_bits =
      (est_dc_gain_bits > 58.0)
          ? 0.0
          : std::max(8.0, 50.0 - 0.70 * std::max(0.0, est_dc_gain_bits - 16.0));

  struct SlopeSample {
    float slope;
    double weight;
  };
  std::vector<SlopeSample> sx_samples;
  std::vector<SlopeSample> sb_samples;
  sx_samples.reserve(num_tiles);
  sb_samples.reserve(num_tiles);
  std::vector<uint8_t> tile_active(num_tiles, 0);
  std::vector<float> tile_sx(num_tiles, 0.0f);
  std::vector<float> tile_sb(num_tiles, jxl::cms::kYToBRatio);
  std::vector<double> tile_wx(num_tiles, 0.0);
  std::vector<double> tile_wb(num_tiles, 0.0);

  const double chroma_w =
      2.0 - jxl::Clamp1((butteraugli_distance - 2.25f) * (1.0f / 1.75f), 0.0f,
                        1.0f);
  constexpr double kLambda = 0.08086717 * 0.25 * 1.375;
  const double rd_mul_x = (0.30 * chroma_w) / kLambda;
  const double rd_mul_b = (0.28 * chroma_w) / kLambda;

  for (size_t ti = 0; ti < num_tiles; ++ti) {
    if (tile_blks[ti] == 0) continue;
    const double q_y = tile_qy[ti] / static_cast<double>(tile_blks[ti]);
    const double q_x = q_y * x_qm_multiplier * qm_x_avg;
    const double q_b = q_y * b_qm_multiplier * qm_b_avg;
    const double wx = tile_yy[ti] * q_x * q_x * rd_mul_x;
    const double wb = tile_yy[ti] * q_b * q_b * rd_mul_b;
    tile_wx[ti] = wx;
    tile_wb[ti] = wb;
    if (tile_yy[ti] > 1e-7 && (wx + wb) > 0.25) {
      tile_active[ti] = 1;
      const float sx = jxl::Clamp1(
          static_cast<float>(tile_yx[ti] / tile_yy[ti]), -2.5f, 2.5f);
      const float sb = jxl::Clamp1(
          static_cast<float>(tile_yb[ti] / tile_yy[ti]), -1.5f, 3.5f);
      tile_sx[ti] = sx;
      tile_sb[ti] = sb;
      sx_samples.push_back({sx, wx + 0.1});
      sb_samples.push_back({sb, wb + 0.1});
    }
  }
  if (sx_samples.size() < 2) return;

  auto weighted_quantile = [](std::vector<SlopeSample>& v, float q) -> float {
    std::sort(v.begin(), v.end(),
              [](const SlopeSample& a, const SlopeSample& b) {
                return a.slope < b.slope;
              });
    double tot = 0.0;
    for (const auto& s : v) tot += s.weight;
    const double target = tot * q;
    double acc = 0.0;
    for (const auto& s : v) {
      acc += s.weight;
      if (acc >= target) return s.slope;
    }
    return v.back().slope;
  };

  const float raw_med_x = weighted_quantile(sx_samples, 0.50f);
  const float raw_med_b = weighted_quantile(sb_samples, 0.50f);
  const float raw_mid_x = 0.5f * (weighted_quantile(sx_samples, 0.05f) +
                                  weighted_quantile(sx_samples, 0.95f));
  const float raw_mid_b = 0.5f * (weighted_quantile(sb_samples, 0.05f) +
                                  weighted_quantile(sb_samples, 0.95f));
  const float dc_est_x = static_cast<float>(dc_num_x / dc_den);
  const float dc_est_b =
      jxl::cms::kYToBRatio + static_cast<float>(dc_num_b / dc_den);

  struct Candidate {
    uint32_t cf;
    float bx;
    float bb;
    int32_t neutral_x;
    int32_t neutral_b;
    double extra_hdr;
  };
  auto make_cand = [&](uint32_t cf, float rx, float rb, double hdr) {
    const float fcf = static_cast<float>(cf);
    const float inv_cf = 1.0f / fcf;
    // At d > 1.25 keep neutral (0, kYToBRatio) reachable within [-126, 126] so
    // smooth-region DC CfL rejection always restores neutral DC correlation;
    // at ultra-fine d <= 1.25 allow full [-1.5, 1.5] centering while keeping
    // the frame's own DC correlation slope reachable within [-120, 120].
    const int32_t max_m =
        (butteraugli_distance > 2.25f)
            ? 126
            : static_cast<int32_t>(std::lround(1.5f * fcf));
    int32_t mx = jxl::Clamp1<int32_t>(std::lround(rx * fcf), -max_m, max_m);
    int32_t mb = jxl::Clamp1<int32_t>(
        std::lround((rb - jxl::cms::kYToBRatio) * fcf), -max_m, max_m);
    if (butteraugli_distance <= 2.25f) {
      const int32_t dc_mx = static_cast<int32_t>(std::lround(dc_est_x * fcf));
      const int32_t dc_mb = static_cast<int32_t>(
          std::lround((dc_est_b - jxl::cms::kYToBRatio) * fcf));
      mx = jxl::Clamp1<int32_t>(mx, dc_mx - 120, dc_mx + 120);
      mb = jxl::Clamp1<int32_t>(mb, dc_mb - 120, dc_mb + 120);
    }
    const float bx = RoundToF16Exact(static_cast<float>(mx) * inv_cf);
    const float bb = RoundToF16Exact(jxl::cms::kYToBRatio +
                                     static_cast<float>(mb) * inv_cf);
    const int32_t nx =
        jxl::Clamp1<int32_t>(std::lround((0.0f - bx) * fcf), -128, 127);
    const int32_t nb = jxl::Clamp1<int32_t>(
        std::lround((jxl::cms::kYToBRatio - bb) * fcf), -128, 127);
    return Candidate{cf, bx, bb, nx, nb, hdr};
  };

  const Candidate cands[] = {
      {kDefaultColorFactor, 0.0f, jxl::cms::kYToBRatio, 0, 0, 0.0},
      make_cand(kDefaultColorFactor, raw_med_x, raw_med_b, eff_header_bits),
      make_cand(kDefaultColorFactor, raw_mid_x, raw_mid_b, eff_header_bits),
      make_cand(64u, raw_med_x, raw_med_b, eff_header_bits + 12.0),
      make_cand(128u, raw_med_x, raw_med_b, eff_header_bits + 8.0),
      make_cand(128u, raw_mid_x, raw_mid_b, eff_header_bits + 8.0),
      {256u, 0.0f, jxl::cms::kYToBRatio, 0, 0, eff_header_bits},
      make_cand(256u, raw_med_x, raw_med_b, eff_header_bits),
      make_cand(256u, raw_mid_x, raw_mid_b, eff_header_bits),
  };

  std::vector<int32_t> qx_grid(num_tiles, 0);
  std::vector<int32_t> qb_grid(num_tiles, 0);
  auto eval_cand = [&](const Candidate& c) -> double {
    const float inv_cf = 1.0f / static_cast<float>(c.cf);
    double ac_err_bits = 0.0;
    for (size_t ti = 0; ti < num_tiles; ++ti) {
      if (!tile_active[ti]) {
        qx_grid[ti] = 0;
        qb_grid[ti] = 0;
        continue;
      }
      const int32_t qx = jxl::Clamp1<int32_t>(
          std::lround((tile_sx[ti] - c.bx) * static_cast<float>(c.cf)), -128,
          127);
      const int32_t qb = jxl::Clamp1<int32_t>(
          std::lround((tile_sb[ti] - c.bb) * static_cast<float>(c.cf)), -128,
          127);
      qx_grid[ti] = qx;
      qb_grid[ti] = qb;
      const double rx = static_cast<double>(c.bx + qx * inv_cf) - tile_sx[ti];
      const double rb = static_cast<double>(c.bb + qb * inv_cf) - tile_sb[ti];
      ac_err_bits += tile_wx[ti] * rx * rx + tile_wb[ti] * rb * rb;
    }
    double map_bits = 0.0;
    for (size_t ty = 0; ty < ty_num; ++ty) {
      for (size_t tx = 0; tx < tx_num; ++tx) {
        const size_t ti = ty * tx_num + tx;
        auto pred_2d = [&](const std::vector<int32_t>& g) -> int32_t {
          if (tx == 0 && ty == 0) return 0;
          if (tx == 0) return g[(ty - 1) * tx_num + tx];
          if (ty == 0) return g[ty * tx_num + (tx - 1)];
          const int32_t l = g[ty * tx_num + (tx - 1)];
          const int32_t t = g[(ty - 1) * tx_num + tx];
          const int32_t tl = g[(ty - 1) * tx_num + (tx - 1)];
          return std::min(std::max(l, t), std::max(std::min(l, t), l + t - tl));
        };
        map_bits += CmapResidualBitsScalar(qx_grid[ti] - pred_2d(qx_grid));
        map_bits += CmapResidualBitsScalar(qb_grid[ti] - pred_2d(qb_grid));
      }
    }
    return c.extra_hdr + map_bits + ac_err_bits;
  };

  const double default_cost = eval_cand(cands[0]);
  const double req_margin = 14.0;
  double best_cost = default_cost - req_margin;
  size_t best_idx = 0;
  for (size_t i = 1; i < sizeof(cands) / sizeof(cands[0]); ++i) {
    if (cands[i].cf < kDefaultColorFactor && butteraugli_distance < 2.4f) {
      continue;
    }
    if (cands[i].cf > kDefaultColorFactor && butteraugli_distance > 2.2f) {
      continue;
    }
    const double cost = eval_cand(cands[i]);
    if (cost < best_cost) {
      best_cost = cost;
      best_idx = i;
    }
  }

  if (best_idx != 0) {
    cmap->base().SetColorFactor(cands[best_idx].cf);
    cmap->base().SetBaseCorrelationX(cands[best_idx].bx);
    cmap->base().SetBaseCorrelationB(cands[best_idx].bb);
    cmap->base().SetYToXDC(cands[best_idx].neutral_x);
    cmap->base().SetYToBDC(cands[best_idx].neutral_b);
  }
}

Status CfLHeuristics::ComputeTile(
    const Rect& r, const Image3F& opsin, const Rect& opsin_rect,
    const DequantMatrices& dequant, const AcStrategyImage* ac_strategy,
    const ImageI* raw_quant_field, const Quantizer* quantizer, bool fast,
    size_t thread, ColorCorrelationMap* cmap, bool shrink_towards_zero,
    float x_qm_multiplier, float b_qm_multiplier, float butteraugli_distance) {
  bool use_dct8 = ac_strategy == nullptr;
  Span<float> scratch(mem.address<float>() + thread * ItemsPerThread(),
                      ItemsPerThread());
  return HWY_DYNAMIC_DISPATCH(ComputeTile)(
      opsin, opsin_rect, dequant, ac_strategy, raw_quant_field, quantizer, r,
      fast, use_dct8, shrink_towards_zero, x_qm_multiplier, b_qm_multiplier,
      butteraugli_distance, cmap->base().GetColorFactor(),
      cmap->base().GetBaseCorrelationX(), cmap->base().GetBaseCorrelationB(),
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
