// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_gaborish.h"

#include <jxl/memory_manager.h>

#include <hwy/base.h>

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/convolve.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_ops.h"

namespace jxl {
namespace {

void SolveSymmetric5x5GaborishInverse(double w1, double w2, float out_rel[5]) {
  static constexpr float kDefaultGab[5] = {
      -0.09495815671340026f, -0.041031725066768575f, 0.013710004822696948f,
      0.006510206083837737f, -0.0014789063378272242f,
  };
  if (w1 + w2 < 1e-6) {
    for (int k = 0; k < 5; ++k) out_rel[k] = 0.0f;
    return;
  }
  const double z = 1.0 + 4.0 * (w1 + w2);
  const double g3[3][3] = {
      {w2 / z, w1 / z, w2 / z},
      {w1 / z, 1.0 / z, w1 / z},
      {w2 / z, w1 / z, w2 / z},
  };
  static constexpr int kMult[5] = {4, 4, 4, 8, 4};
  auto orbit_idx = [](int i, int j) -> int {
    int ai = (i < 0) ? -i : i;
    int aj = (j < 0) ? -j : j;
    if (ai > aj) {
      const int t = ai;
      ai = aj;
      aj = t;
    }
    if (ai == 0 && aj == 0) return 0;
    if (ai == 0 && aj == 1) return 1;
    if (ai == 1 && aj == 1) return 2;
    if (ai == 0 && aj == 2) return 3;
    if (ai == 1 && aj == 2) return 4;
    return 5;
  };
  double c0[49] = {};
  double a_col[5][49] = {};
  for (int i = -2; i <= 2; ++i) {
    for (int j = -2; j <= 2; ++j) {
      const int orb = orbit_idx(i, j);
      for (int u = 0; u < 3; ++u) {
        for (int v = 0; v < 3; ++v) {
          const int idx7 = (i + 2 + u) * 7 + (j + 2 + v);
          const double gv = g3[u][v];
          if (orb == 0) {
            c0[idx7] += gv;
          } else {
            a_col[orb - 1][idx7] += gv;
          }
        }
      }
    }
  }
  for (int k = 0; k < 5; ++k) {
    for (int p = 0; p < 49; ++p) {
      a_col[k][p] -= static_cast<double>(kMult[k]) * c0[p];
    }
  }
  double ata[5][6] = {};
  for (int r = 0; r < 5; ++r) {
    for (int c = 0; c < 5; ++c) {
      double s = 0.0;
      for (int p = 0; p < 49; ++p) s += a_col[r][p] * a_col[c][p];
      ata[r][c] = s + ((r == c) ? 1e-12 : 0.0);
    }
    double rhs = 0.0;
    for (int p = 0; p < 49; ++p) {
      const double tgt = (p == 24) ? 1.0 : 0.0;
      rhs += a_col[r][p] * (tgt - c0[p]);
    }
    ata[r][5] = rhs;
  }
  for (int col = 0; col < 5; ++col) {
    int piv = col;
    for (int r = col + 1; r < 5; ++r) {
      if (std::abs(ata[r][col]) > std::abs(ata[piv][col])) piv = r;
    }
    if (std::abs(ata[piv][col]) < 1e-15) {
      for (int k = 0; k < 5; ++k) out_rel[k] = kDefaultGab[k];
      return;
    }
    if (piv != col) {
      for (int c = col; c <= 5; ++c) std::swap(ata[col][c], ata[piv][c]);
    }
    const double inv_d = 1.0 / ata[col][col];
    for (int c = col; c <= 5; ++c) ata[col][c] *= inv_d;
    for (int r = 0; r < 5; ++r) {
      if (r == col) continue;
      const double f = ata[r][col];
      for (int c = col; c <= 5; ++c) ata[r][c] -= f * ata[col][c];
    }
  }
  double x0 = 1.0;
  for (int k = 0; k < 5; ++k) {
    x0 -= static_cast<double>(kMult[k]) * ata[k][5];
  }
  if (x0 < 1e-4) {
    for (int k = 0; k < 5; ++k) out_rel[k] = kDefaultGab[k];
    return;
  }
  for (int k = 0; k < 5; ++k) {
    out_rel[k] = static_cast<float>(ata[k][5] / x0);
  }
}

}  // namespace

Status GaborishInverse(Image3F* in_out, const Rect& rect, const float w1[3],
                       const float w2[3], const float mul[3],
                       ThreadPool* pool) {
  JxlMemoryManager* memory_manager = in_out->memory_manager();
  WeightsSymmetric5 weights[3];
  static const float kGaborish[5] = {
      -0.09495815671340026f, -0.041031725066768575f, 0.013710004822696948f,
      0.006510206083837737f, -0.0014789063378272242f,
  };
  constexpr float kStdW1 = 1.1f * 0.104699568f;
  constexpr float kStdW2 = 1.1f * 0.055680538f;
  for (int i = 0; i < 3; ++i) {
    float g[5];
    if (w1 != nullptr && w2 != nullptr &&
        (std::abs(w1[i] - kStdW1) > 1e-5f ||
         std::abs(w2[i] - kStdW2) > 1e-5f)) {
      SolveSymmetric5x5GaborishInverse(w1[i], w2[i], g);
    } else {
      for (int k = 0; k < 5; ++k) g[k] = kGaborish[k];
    }
    double sum =
        1.0 + mul[i] * 4 * (g[0] + g[1] + g[2] + g[4] + 2 * g[3]);
    if (sum < 1e-5) {
      sum = 1e-5;
    }
    const float normalize = static_cast<float>(1.0 / sum);
    const float normalize_mul = mul[i] * normalize;
    weights[i] = WeightsSymmetric5{{HWY_REP4(normalize)},
                                   {HWY_REP4(normalize_mul * g[0])},
                                   {HWY_REP4(normalize_mul * g[2])},
                                   {HWY_REP4(normalize_mul * g[1])},
                                   {HWY_REP4(normalize_mul * g[4])},
                                   {HWY_REP4(normalize_mul * g[3])}};
  }
  ImageF temp;
  JXL_ASSIGN_OR_RETURN(temp,
                       ImageF::Create(memory_manager, in_out->Plane(2).xsize(),
                                      in_out->Plane(2).ysize()));
  JXL_RETURN_IF_ERROR(CopyImageTo(in_out->Plane(2), &temp));
  Rect xrect = rect.Extend(3, Rect(*in_out));
  JXL_RETURN_IF_ERROR(Symmetric5(in_out->Plane(0), xrect, weights[0], pool,
                                 &in_out->Plane(2), xrect));
  JXL_RETURN_IF_ERROR(Symmetric5(in_out->Plane(1), xrect, weights[1], pool,
                                 &in_out->Plane(0), xrect));
  JXL_RETURN_IF_ERROR(
      Symmetric5(temp, xrect, weights[2], pool, &in_out->Plane(1), xrect));
  in_out->Plane(0).Swap(in_out->Plane(1));
  in_out->Plane(0).Swap(in_out->Plane(2));
  return true;
}

Status GaborishInverse(Image3F* in_out, const Rect& rect, const float mul[3],
                       ThreadPool* pool) {
  return GaborishInverse(in_out, rect, nullptr, nullptr, mul, pool);
}

}  // namespace jxl
