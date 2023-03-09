// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This C++ example decodes a JPEG XL image in one shot (all input bytes
// available at once). The example outputs the pixels and color information to a
// floating point image and an ICC profile on disk.

#include <inttypes.h>
#include <jxl/decode.h>
#include <jxl/decode_cxx.h>
#include <jxl/resizable_parallel_runner.h>
#include <jxl/resizable_parallel_runner_cxx.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK(x)                               \
  if (!(x)) {                                  \
    fprintf(stderr, "check failed: %s\n", #x); \
    abort();                                   \
  }

struct YUVP010Video {
  std::vector<std::vector<uint16_t>> frames;
  size_t xsize, ysize;
};

YUVP010Video DecodeJpegXlOneShot(const uint8_t* jxl, size_t size) {
  // Multi-threaded parallel runner.
  auto runner = JxlResizableParallelRunnerMake(nullptr);

  YUVP010Video ret;

  auto dec = JxlDecoderMake(nullptr);
  CHECK(JXL_DEC_SUCCESS ==
        JxlDecoderSubscribeEvents(dec.get(),
                                  JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE));

  CHECK(JXL_DEC_SUCCESS ==
        JxlDecoderSetParallelRunner(dec.get(), JxlResizableParallelRunner,
                                    runner.get()));

  JxlBasicInfo info;
  JxlPixelFormat format = {4, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};

  JxlDecoderSetInput(dec.get(), jxl, size);
  JxlDecoderCloseInput(dec.get());

  std::vector<float> pixels;

  for (;;) {
    JxlDecoderStatus status = JxlDecoderProcessInput(dec.get());

    CHECK(status != JXL_DEC_ERROR);
    CHECK(status != JXL_DEC_NEED_MORE_INPUT);

    if (status == JXL_DEC_BASIC_INFO) {
      CHECK(JXL_DEC_SUCCESS == JxlDecoderGetBasicInfo(dec.get(), &info));
      ret.xsize = info.xsize;
      ret.ysize = info.ysize;
      pixels.resize(4 * ret.xsize * ret.ysize);
      JxlResizableParallelRunnerSetThreads(
          runner.get(),
          JxlResizableParallelRunnerSuggestThreads(info.xsize, info.ysize));
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      size_t buffer_size;
      CHECK(JXL_DEC_SUCCESS ==
            JxlDecoderImageOutBufferSize(dec.get(), &format, &buffer_size));
      CHECK(buffer_size == ret.xsize * ret.ysize * 16);
      void* pixels_buffer = (void*)pixels.data();
      size_t pixels_buffer_size = pixels.size() * sizeof(float);
      CHECK(JXL_DEC_SUCCESS == JxlDecoderSetImageOutBuffer(dec.get(), &format,
                                                           pixels_buffer,
                                                           pixels_buffer_size));
    } else if (status == JXL_DEC_FULL_IMAGE) {
      // YCbCr conversion.
      auto cvt = [](float f) {
        return static_cast<uint16_t>(std::round(
                   std::max(0.0f, std::min(65535.0f, f * 65535.0f)))) &
               0xFFC0;
      };
      ret.frames.emplace_back(ret.xsize * ret.ysize * 3 / 2);
      for (size_t y = 0; y < ret.ysize; y += 2) {
        for (size_t x = 0; x < ret.xsize; x += 2) {
          float Y[4];
          float Cb[4];
          float Cr[4];
          for (size_t i = 0; i < 4; i++) {
            size_t yy = y + (i % 2);
            size_t xx = x + (i / 2);
            float r = pixels[4 * yy * info.xsize + 4 * xx + 0];
            float g = pixels[4 * yy * info.xsize + 4 * xx + 1];
            float b = pixels[4 * yy * info.xsize + 4 * xx + 2];
            Y[i] = 0.299 * r + 0.587 * g + 0.114 * b;
            Cb[i] = -0.168736 * r - 0.331264 * g + 0.5 * b;
            Cr[i] = 0.5 * r - 0.418688 * g - 0.081312 * b;
          }
          float Cb420 =
              (Cb[0] + Cb[1] + Cb[2] + Cb[3]) * 0.25 + (1024.0 / 2047.0);
          float Cr420 =
              (Cr[0] + Cr[1] + Cr[2] + Cr[3]) * 0.25 + (1024.0 / 2047.0);
          for (size_t i = 0; i < 4; i++) {
            size_t yy = y + (i % 2);
            size_t xx = x + (i / 2);
            ret.frames.back()[ret.xsize * yy + xx] = cvt(Y[i]);
          }
          ret.frames.back()[ret.xsize * ret.ysize + ret.xsize * (y / 2) + x] =
              cvt(Cb420);
          ret.frames
              .back()[ret.xsize * ret.ysize + ret.xsize * (y / 2) + x + 1] =
              cvt(Cr420);
        }
      }
    } else if (status == JXL_DEC_SUCCESS) {
      return ret;
    } else {
      CHECK(false);
    }
  }
}

bool LoadFile(const char* filename, std::vector<uint8_t>* out) {
  FILE* file = fopen(filename, "rb");
  if (!file) {
    return false;
  }

  if (fseek(file, 0, SEEK_END) != 0) {
    fclose(file);
    return false;
  }

  long size = ftell(file);
  // Avoid invalid file or directory.
  if (size >= LONG_MAX || size < 0) {
    fclose(file);
    return false;
  }

  if (fseek(file, 0, SEEK_SET) != 0) {
    fclose(file);
    return false;
  }

  out->resize(size);
  size_t readsize = fread(out->data(), 1, size, file);
  if (fclose(file) != 0) {
    return false;
  }

  return readsize == static_cast<size_t>(size);
}

bool WriteFile(const char* filename, const uint8_t* data, size_t size) {
  FILE* file = fopen(filename, "wb");
  if (!file) {
    fprintf(stderr, "Could not open %s for writing", filename);
    return false;
  }
  fwrite(data, 1, size, file);
  if (fclose(file) != 0) {
    return false;
  }
  return true;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    fprintf(stderr,
            "Usage: %s <jxl> <output_base>\n"
            "Where:\n"
            "  jxl = input JPEG XL image filename\n"
            "  output_base = basename for .bin files\n"
            "Output files will be overwritten.\n",
            argv[0]);
    return 1;
  }

  const char* jxl_filename = argv[1];
  const char* out_basename = argv[2];

  std::vector<uint8_t> jxl;
  if (!LoadFile(jxl_filename, &jxl)) {
    fprintf(stderr, "couldn't load %s\n", jxl_filename);
    return 1;
  }

  YUVP010Video out = DecodeJpegXlOneShot(jxl.data(), jxl.size());
  for (size_t i = 0; i < out.frames.size(); i++) {
    CHECK(WriteFile((out_basename + std::to_string(i) + ".bin").c_str(),
                    (const uint8_t*)out.frames[i].data(),
                    out.frames[i].size() * 2));
  }
  return 0;
}
