// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/modular/transform/squeeze.h"

#include <stdlib.h>

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/common.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/transform/transform.h"
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/modular/transform/squeeze.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// Equivalent to SmoothTendency(), but without branches
// Only faster when SIMD can be used
inline pixel_type SmoothTendencyNoBranch(pixel_type B, pixel_type a,
                                         pixel_type n) {
  pixel_type Ba = B - a;
  pixel_type an = a - n;
  pixel_type nonmono = Ba ^ an;
  pixel_type absBa = std::abs(Ba);
  pixel_type absan = std::abs(an);
  pixel_type absBn = std::abs(B - n);
  pixel_type absdiff = (absBa / 3 + absBn + 2) / 4;
  pixel_type skipdiff = Ba != 0;
  skipdiff &= an != 0;
  skipdiff &= nonmono < 0;
  pixel_type absBa2 = absBa * 2 + (absdiff & 1);
  absdiff = (absdiff > absBa2 ? absBa * 2 + 1 : absdiff);
  pixel_type absan2 = absan * 2;
  absdiff = (absdiff + (absdiff & 1) > absan2 ? absan2 : absdiff);
  pixel_type diff = (B < n ? -absdiff : absdiff);
  diff = (skipdiff ? 0 : diff);
  return diff;
}

template <size_t count>
JXL_INLINE void fast_unsqueeze(const pixel_type *p_residual,
                               const pixel_type *p_avg,
                               const pixel_type *p_navg,
                               const pixel_type *p_pout,
                               pixel_type *JXL_RESTRICT p_out,
                               pixel_type *JXL_RESTRICT p_nout) {
#pragma clang loop vectorize(enable)
  for (size_t x = 0; x < count; x++) {
    pixel_type avg = p_avg[x];
    pixel_type next_avg = p_navg[x];
    pixel_type top = p_pout[x];
    pixel_type tendency = SmoothTendencyNoBranch(top, avg, next_avg);

    pixel_type diff_minus_tendency = p_residual[x];
    pixel_type diff = diff_minus_tendency + tendency;

    pixel_type out =
        ((avg * 2) + diff + (diff < 0 ? (diff & 1) : -(diff & 1))) >> 1;

    p_out[x] = out;
    p_nout[x] = out - diff;
  }
}

Status InvHSqueeze(Image &input, uint32_t c, uint32_t rc, ThreadPool *pool) {
  JXL_ASSERT(c < input.channel.size());
  JXL_ASSERT(rc < input.channel.size());
  Channel &chin = input.channel[c];
  const Channel &chin_residual = input.channel[rc];
  // These must be valid since we ran MetaApply already.
  JXL_ASSERT(chin.w == DivCeil(chin.w + chin_residual.w, 2));
  JXL_ASSERT(chin.h == chin_residual.h);

  if (chin_residual.w == 0) {
    // Short-circuit: output channel has same dimensions as input.
    input.channel[c].hshift--;
    return true;
  }

  // Note: chin.w >= chin_residual.w and at most 1 different.
  Channel chout(chin.w + chin_residual.w, chin.h, chin.hshift - 1, chin.vshift);
  JXL_DEBUG_V(4,
              "Undoing horizontal squeeze of channel %i using residuals in "
              "channel %i (going from width %" PRIuS " to %" PRIuS ")",
              c, rc, chin.w, chout.w);

  if (chin_residual.h == 0) {
    // Short-circuit: channel with no pixels.
    input.channel[c] = std::move(chout);
    return true;
  }

#if HWY_TARGET != HWY_SCALAR

  // somewhat complicated trickery just to be able to SIMD this
  intptr_t onerow_in = chin.plane.PixelsPerRow();
  intptr_t onerow_inr = chin_residual.plane.PixelsPerRow();
  intptr_t onerow_out = chout.plane.PixelsPerRow();
  constexpr int kRowsPerThread = 8;
  JXL_RETURN_IF_ERROR(RunOnPool(
      pool, 0, chin.h / kRowsPerThread, ThreadPool::NoInit,
      [&](const uint32_t task, size_t /* thread */) {
        const size_t y0 = task * kRowsPerThread;
        const pixel_type *JXL_RESTRICT p_residual = chin_residual.Row(y0);
        const pixel_type *JXL_RESTRICT p_avg = chin.Row(y0);
        pixel_type *JXL_RESTRICT p_out = chout.Row(y0);

        pixel_type b_p_avg[9][kRowsPerThread];
        pixel_type b_p_residual[8][kRowsPerThread];
        pixel_type b_p_out_even[8][kRowsPerThread];
        pixel_type b_p_out_odd[8][kRowsPerThread];
        size_t x = 0;
        if (chin_residual.w > 16)
          for (; x < ((chin_residual.w - 9) & (~7)); x++) {
            if ((x & 7) == 0) {
              for (size_t y = 0; y < kRowsPerThread; y++) {
                for (size_t xx = x; xx < x + 8; xx++) {
                  b_p_residual[xx & 7][y] = p_residual[xx + onerow_inr * y];
                  b_p_avg[xx & 7][y] = p_avg[xx + onerow_in * y];
                }
                b_p_avg[8][y] = p_avg[x + 8 + onerow_in * y];
              }
            }
            fast_unsqueeze<kRowsPerThread>(
                b_p_residual[x & 7], b_p_avg[x & 7], b_p_avg[(x & 7) + 1],
                (x ? b_p_out_odd[(x - 1) & 7] : b_p_avg[x & 7]),
                b_p_out_even[x & 7], b_p_out_odd[x & 7]);

            if ((x & 7) == 7) {
              for (size_t y = 0; y < kRowsPerThread; y++) {
                for (size_t xx = x - 7; xx <= x; xx++) {
                  p_out[(xx << 1) + onerow_out * y] = b_p_out_even[xx & 7][y];
                  p_out[(xx << 1) + 1 + onerow_out * y] =
                      b_p_out_odd[xx & 7][y];
                }
              }
            }
          }
        size_t x0 = x;
        for (size_t y = 0; y < kRowsPerThread; y++) {
          for (x = x0; x < chin_residual.w; x++) {
            pixel_type diff_minus_tendency = p_residual[x + onerow_inr * y];
            pixel_type avg = p_avg[x + onerow_in * y];
            pixel_type next_avg =
                (x + 1 < chin.w ? p_avg[x + 1 + onerow_in * y] : avg);
            pixel_type left = (x ? p_out[(x << 1) - 1 + onerow_out * y] : avg);
            pixel_type tendency = SmoothTendency(left, avg, next_avg);
            pixel_type diff = diff_minus_tendency + tendency;
            pixel_type A =
                ((avg * 2) + diff + (diff > 0 ? -(diff & 1) : (diff & 1))) >> 1;
            p_out[(x << 1) + onerow_out * y] = A;
            pixel_type B = A - diff;
            p_out[(x << 1) + 1 + onerow_out * y] = B;
          }
          if (chout.w & 1)
            p_out[chout.w - 1 + onerow_out * y] =
                p_avg[chin.w - 1 + onerow_in * y];
        }
      },
      "InvHorizontalSqueeze"));
  size_t firstrow = chin.h / kRowsPerThread * kRowsPerThread;
#else
  size_t firstrow = 0;
#endif

  JXL_RETURN_IF_ERROR(RunOnPool(
      pool, firstrow, chin.h, ThreadPool::NoInit,
      [&](const uint32_t task, size_t /* thread */) {
        const size_t y = task;
        const pixel_type *JXL_RESTRICT p_residual = chin_residual.Row(y);
        const pixel_type *JXL_RESTRICT p_avg = chin.Row(y);
        pixel_type *JXL_RESTRICT p_out = chout.Row(y);

        // special case for x=0 so we don't have to check x>0
        pixel_type avg = p_avg[0];
        pixel_type next_avg = (1 < chin.w ? p_avg[1] : avg);
        pixel_type tendency = SmoothTendency(avg, avg, next_avg);
        pixel_type diff = p_residual[0] + tendency;
        pixel_type A =
            ((avg * 2) + diff + (diff > 0 ? -(diff & 1) : (diff & 1))) >> 1;
        pixel_type B = A - diff;
        p_out[0] = A;
        p_out[1] = B;

        for (size_t x = 1; x < chin_residual.w; x++) {
          pixel_type diff_minus_tendency = p_residual[x];
          pixel_type avg = p_avg[x];
          pixel_type next_avg = (x + 1 < chin.w ? p_avg[x + 1] : avg);
          pixel_type left = p_out[(x << 1) - 1];
          pixel_type tendency = SmoothTendency(left, avg, next_avg);
          pixel_type diff = diff_minus_tendency + tendency;
          pixel_type A =
              ((avg * 2) + diff + (diff > 0 ? -(diff & 1) : (diff & 1))) >> 1;
          p_out[x << 1] = A;
          pixel_type B = A - diff;
          p_out[(x << 1) + 1] = B;
        }
        if (chout.w & 1) p_out[chout.w - 1] = p_avg[chin.w - 1];
      },
      "InvHorizontalSqueeze"));
  input.channel[c] = std::move(chout);
  return true;
}

Status InvVSqueeze(Image &input, uint32_t c, uint32_t rc, ThreadPool *pool) {
  JXL_ASSERT(c < input.channel.size());
  JXL_ASSERT(rc < input.channel.size());
  const Channel &chin = input.channel[c];
  const Channel &chin_residual = input.channel[rc];
  // These must be valid since we ran MetaApply already.
  JXL_ASSERT(chin.h == DivCeil(chin.h + chin_residual.h, 2));
  JXL_ASSERT(chin.w == chin_residual.w);

  if (chin_residual.h == 0) {
    // Short-circuit: output channel has same dimensions as input.
    input.channel[c].vshift--;
    return true;
  }

  // Note: chin.h >= chin_residual.h and at most 1 different.
  Channel chout(chin.w, chin.h + chin_residual.h, chin.hshift, chin.vshift - 1);
  JXL_DEBUG_V(
      4,
      "Undoing vertical squeeze of channel %i using residuals in channel "
      "%i (going from height %" PRIuS " to %" PRIuS ")",
      c, rc, chin.h, chout.h);

  if (chin_residual.w == 0) {
    // Short-circuit: channel with no pixels.
    input.channel[c] = std::move(chout);
    return true;
  }

  constexpr int kColsPerThread = 64;
  JXL_RETURN_IF_ERROR(RunOnPool(
      pool, 0, DivCeil(chin.w, kColsPerThread), ThreadPool::NoInit,
      [&](const uint32_t task, size_t /* thread */) {
        const size_t x0 = task * kColsPerThread;
        const size_t x1 = std::min((size_t)(task + 1) * kColsPerThread, chin.w);
        const size_t w = x1 - x0;
        // We only iterate up to std::min(chin_residual.h, chin.h) which is
        // always chin_residual.h.
        for (size_t y = 0; y < chin_residual.h; y++) {
          const pixel_type *JXL_RESTRICT p_residual = chin_residual.Row(y) + x0;
          const pixel_type *JXL_RESTRICT p_avg = chin.Row(y) + x0;
          const pixel_type *JXL_RESTRICT p_navg =
              chin.Row(y + 1 < chin.h ? y + 1 : y) + x0;
          pixel_type *JXL_RESTRICT p_out = chout.Row(y << 1) + x0;
          pixel_type *JXL_RESTRICT p_nout = chout.Row((y << 1) + 1) + x0;
          const pixel_type *p_pout =
              y > 0 ? chout.Row((y << 1) - 1) + x0 : p_avg;
          size_t x = 0;
          for (; x + 7 < w; x += 8) {
            fast_unsqueeze<8>(p_residual + x, p_avg + x, p_navg + x, p_pout + x,
                              p_out + x, p_nout + x);
          }
          for (; x < w; x++) {
            pixel_type avg = p_avg[x];
            pixel_type next_avg = p_navg[x];
            pixel_type top = p_pout[x];
            pixel_type tendency = SmoothTendency(top, avg, next_avg);
            pixel_type diff_minus_tendency = p_residual[x];
            pixel_type diff = diff_minus_tendency + tendency;
            pixel_type out =
                ((avg * 2) + diff + (diff < 0 ? (diff & 1) : -(diff & 1))) >> 1;

            p_out[x] = out;
            // If the chin_residual.h == chin.h, the output has an even number
            // of rows so the next line is fine. Otherwise, this loop won't
            // write to the last output row which is handled separately.
            p_nout[x] = out - diff;
          }
        }
      },
      "InvVertSqueeze"));

  if (chout.h & 1) {
    size_t y = chin.h - 1;
    const pixel_type *p_avg = chin.Row(y);
    pixel_type *p_out = chout.Row(y << 1);
    for (size_t x = 0; x < chin.w; x++) {
      p_out[x] = p_avg[x];
    }
  }
  input.channel[c] = std::move(chout);
  return true;
}

Status InvSqueeze(Image &input, std::vector<SqueezeParams> parameters,
                  ThreadPool *pool) {
  for (int i = parameters.size() - 1; i >= 0; i--) {
    JXL_RETURN_IF_ERROR(
        CheckMetaSqueezeParams(parameters[i], input.channel.size()));
    bool horizontal = parameters[i].horizontal;
    bool in_place = parameters[i].in_place;
    uint32_t beginc = parameters[i].begin_c;
    uint32_t endc = parameters[i].begin_c + parameters[i].num_c - 1;
    uint32_t offset;
    if (in_place) {
      offset = endc + 1;
    } else {
      offset = input.channel.size() + beginc - endc - 1;
    }
    if (beginc < input.nb_meta_channels) {
      // This is checked in MetaSqueeze.
      JXL_ASSERT(input.nb_meta_channels > parameters[i].num_c);
      input.nb_meta_channels -= parameters[i].num_c;
    }

    for (uint32_t c = beginc; c <= endc; c++) {
      uint32_t rc = offset + c - beginc;
      // MetaApply should imply that `rc` is within range, otherwise there's a
      // programming bug.
      JXL_ASSERT(rc < input.channel.size());
      if ((input.channel[c].w < input.channel[rc].w) ||
          (input.channel[c].h < input.channel[rc].h)) {
        return JXL_FAILURE("Corrupted squeeze transform");
      }
      if (horizontal) {
        JXL_RETURN_IF_ERROR(InvHSqueeze(input, c, rc, pool));
      } else {
        JXL_RETURN_IF_ERROR(InvVSqueeze(input, c, rc, pool));
      }
    }
    input.channel.erase(input.channel.begin() + offset,
                        input.channel.begin() + offset + (endc - beginc + 1));
  }
  return true;
}

}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace jxl {

HWY_EXPORT(InvSqueeze);
Status InvSqueeze(Image &input, std::vector<SqueezeParams> parameters,
                  ThreadPool *pool) {
  return HWY_DYNAMIC_DISPATCH(InvSqueeze)(input, parameters, pool);
}

void DefaultSqueezeParameters(std::vector<SqueezeParams> *parameters,
                              const Image &image) {
  int nb_channels = image.channel.size() - image.nb_meta_channels;

  parameters->clear();
  size_t w = image.channel[image.nb_meta_channels].w;
  size_t h = image.channel[image.nb_meta_channels].h;
  JXL_DEBUG_V(
      7, "Default squeeze parameters for %" PRIuS "x%" PRIuS " image: ", w, h);

  // do horizontal first on wide images; vertical first on tall images
  bool wide = (w > h);

  if (nb_channels > 2 && image.channel[image.nb_meta_channels + 1].w == w &&
      image.channel[image.nb_meta_channels + 1].h == h) {
    // assume channels 1 and 2 are chroma, and can be squeezed first for 4:2:0
    // previews
    JXL_DEBUG_V(7, "(4:2:0 chroma), %" PRIuS "x%" PRIuS " image", w, h);
    SqueezeParams params;
    // horizontal chroma squeeze
    params.horizontal = true;
    params.in_place = false;
    params.begin_c = image.nb_meta_channels + 1;
    params.num_c = 2;
    parameters->push_back(params);
    params.horizontal = false;
    // vertical chroma squeeze
    parameters->push_back(params);
  }
  SqueezeParams params;
  params.begin_c = image.nb_meta_channels;
  params.num_c = nb_channels;
  params.in_place = true;

  if (!wide) {
    if (h > JXL_MAX_FIRST_PREVIEW_SIZE) {
      params.horizontal = false;
      parameters->push_back(params);
      h = (h + 1) / 2;
      JXL_DEBUG_V(7, "Vertical (%" PRIuS "x%" PRIuS "), ", w, h);
    }
  }
  while (w > JXL_MAX_FIRST_PREVIEW_SIZE || h > JXL_MAX_FIRST_PREVIEW_SIZE) {
    if (w > JXL_MAX_FIRST_PREVIEW_SIZE) {
      params.horizontal = true;
      parameters->push_back(params);
      w = (w + 1) / 2;
      JXL_DEBUG_V(7, "Horizontal (%" PRIuS "x%" PRIuS "), ", w, h);
    }
    if (h > JXL_MAX_FIRST_PREVIEW_SIZE) {
      params.horizontal = false;
      parameters->push_back(params);
      h = (h + 1) / 2;
      JXL_DEBUG_V(7, "Vertical (%" PRIuS "x%" PRIuS "), ", w, h);
    }
  }
  JXL_DEBUG_V(7, "that's it");
}

Status CheckMetaSqueezeParams(const SqueezeParams &parameter,
                              int num_channels) {
  int c1 = parameter.begin_c;
  int c2 = parameter.begin_c + parameter.num_c - 1;
  if (c1 < 0 || c1 >= num_channels || c2 < 0 || c2 >= num_channels || c2 < c1) {
    return JXL_FAILURE("Invalid channel range");
  }
  return true;
}

Status MetaSqueeze(Image &image, std::vector<SqueezeParams> *parameters) {
  if (parameters->empty()) {
    DefaultSqueezeParameters(parameters, image);
  }

  for (size_t i = 0; i < parameters->size(); i++) {
    JXL_RETURN_IF_ERROR(
        CheckMetaSqueezeParams((*parameters)[i], image.channel.size()));
    bool horizontal = (*parameters)[i].horizontal;
    bool in_place = (*parameters)[i].in_place;
    uint32_t beginc = (*parameters)[i].begin_c;
    uint32_t endc = (*parameters)[i].begin_c + (*parameters)[i].num_c - 1;

    uint32_t offset;
    if (beginc < image.nb_meta_channels) {
      if (endc >= image.nb_meta_channels) {
        return JXL_FAILURE("Invalid squeeze: mix of meta and nonmeta channels");
      }
      if (!in_place)
        return JXL_FAILURE(
            "Invalid squeeze: meta channels require in-place residuals");
      image.nb_meta_channels += (*parameters)[i].num_c;
    }
    if (in_place) {
      offset = endc + 1;
    } else {
      offset = image.channel.size();
    }
    for (uint32_t c = beginc; c <= endc; c++) {
      if (image.channel[c].hshift > 30 || image.channel[c].vshift > 30) {
        return JXL_FAILURE("Too many squeezes: shift > 30");
      }
      size_t w = image.channel[c].w;
      size_t h = image.channel[c].h;
      if (horizontal) {
        image.channel[c].w = (w + 1) / 2;
        image.channel[c].hshift++;
        w = w - (w + 1) / 2;
      } else {
        image.channel[c].h = (h + 1) / 2;
        image.channel[c].vshift++;
        h = h - (h + 1) / 2;
      }
      image.channel[c].shrink();
      Channel dummy(w, h);
      dummy.hshift = image.channel[c].hshift;
      dummy.vshift = image.channel[c].vshift;

      image.channel.insert(image.channel.begin() + offset + (c - beginc),
                           std::move(dummy));
    }
  }
  return true;
}

}  // namespace jxl

#endif