// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "fmjxl.h"

int main(int argc, char** argv) {
  if (argc < 7) {
    fprintf(stderr,
            "Usage: %s out.jxl w h num_reps num_threads f1_ycbcr10.bin "
            "[f2_ycbcr10.bin [...]] \n",
            argv[0]);
    return 1;
  }
  size_t w = atoi(argv[2]);
  size_t h = atoi(argv[3]);
  assert(w % 16 == 0);
  assert(h % 16 == 0);
  assert(w > 256 || h > 256);
  size_t num_reps = atoi(argv[4]);
  size_t num_threads = atoi(argv[5]);

  assert(num_reps > 0);

  std::vector<std::vector<uint8_t>> input_data;

  for (size_t f = 0; f + 6 < (size_t)argc; f++) {
    FILE* in = fopen(argv[6 + f], "r");
    assert(in);
    fseek(in, 0, SEEK_END);
    ssize_t size = ftell(in);
    fseek(in, 0, SEEK_SET);
    input_data.emplace_back(size, 0);
    fread(input_data.back().data(), 1, size, in);
    fclose(in);
  }

  uint8_t* encoded = nullptr;
  size_t encoded_size = 0;

  auto start = std::chrono::high_resolution_clock::now();
  for (size_t _ = 0; _ < num_reps; _++) {
    free(encoded);
    encoded = nullptr;
    encoded_size = 0;
    auto encoder = FastMJXLCreateEncoder(w, h);
    for (size_t i = 0; i < input_data.size(); i++) {
      FastMJXLAddYCbCrP010Frame(input_data[i].data(),
                                input_data[i].data() + 2 * w * h,
                                i + 1 == input_data.size(), encoder);
      size_t chunk = FastMJXLGetOutputBufferSize(encoder);
      encoded = (uint8_t*)realloc(encoded, encoded_size + chunk);
      memcpy(encoded + encoded_size, FastMJXLGetOutputBuffer(encoder), chunk);
      encoded_size += chunk;
    }
    FastMJXLDestroyEncoder(encoder);
  }
  auto stop = std::chrono::high_resolution_clock::now();

  if (num_reps > 1) {
    float us =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();
    size_t pixels = w * h * num_reps * input_data.size();
    float mps = pixels / us;
    fprintf(stderr, "%10.3f MP/s\n", mps);
    fprintf(stderr, "%10.3f bits/pixel\n",
            encoded_size * 8.0 / float(w) / float(h));
  }

  FILE* out = fopen(argv[1], "w");
  assert(out);
  fwrite(encoded, encoded_size, 1, out);
  fclose(out);
  free(encoded);
}
