// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_MODULAR_TRANSFORM_ENC_PALETTE_H_
#define LIB_JXL_MODULAR_TRANSFORM_ENC_PALETTE_H_

#include <cstdint>

#include <map>
#include <vector>

#include "lib/jxl/base/status.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/options.h"
#include "lib/jxl/modular/transform/transform.h"

namespace jxl {

namespace palette_internal {

void OrderPaletteGreedy(
    const Image &input, uint32_t begin_c, uint32_t nb,
    std::vector<std::vector<pixel_type>> &candidate_palette,
    const std::map<std::vector<pixel_type>, size_t> &color_freq_map);



void OrderPaletteMinLA(
    const Image &input, uint32_t begin_c, uint32_t nb,
    std::vector<std::vector<pixel_type>> &candidate_palette,
    const std::map<std::vector<pixel_type>, size_t> &color_freq_map);

void OrderPaletteMinLAGradient(
    const Image &input, uint32_t begin_c, uint32_t nb,
    std::vector<std::vector<pixel_type>> &candidate_palette,
    const std::map<std::vector<pixel_type>, size_t> &color_freq_map);

}  // namespace palette_internal

Status FwdPalette(Image &input, uint32_t begin_c, uint32_t end_c,
                  uint32_t &nb_colors, uint32_t &nb_deltas,
                  PaletteOrdering ordering, bool lossy, Predictor &predictor,
                  const weighted::Header &wp_header);

}  // namespace jxl

#endif  // LIB_JXL_MODULAR_TRANSFORM_ENC_PALETTE_H_
