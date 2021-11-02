// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdio.h>

#include <numeric>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/fast_dct_test.cc"
#include <hwy/foreach_target.h>

#include "lib/jxl/base/random.h"
#include "lib/jxl/dct-inl.h"
#include "lib/jxl/fast_dct-inl.h"
#include "lib/jxl/transpose-inl.h"

// Test utils
#include <hwy/highway.h>
#include <hwy/tests/test_util-inl.h>
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {
namespace {

template <size_t N, size_t M>
HWY_NOINLINE void TestFastTranspose() {
#if HWY_TARGET == HWY_NEON
  int16_t array[N * M];
  int16_t transposed[N * M];
  std::iota(array, array + N * M, 0);
  for (size_t j = 0; j < 100000000 / (N * M); j++) {
    FastTransposeBlock(array, M, N, M, transposed, N);
  }
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      EXPECT_EQ(array[j * M + i], transposed[i * N + j]);
    }
  }
#endif
}

template <size_t N, size_t M>
HWY_NOINLINE void TestFloatTranspose() {
  float array[N * M];
  float transposed[N * M];
  std::iota(array, array + N * M, 0);
  for (size_t j = 0; j < 100000000 / (N * M); j++) {
    Transpose<N, M>::Run(DCTFrom(array, M), DCTTo(transposed, N));
  }
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      EXPECT_EQ(array[j * M + i], transposed[i * N + j]);
    }
  }
}

template <size_t N, size_t M>
HWY_NOINLINE void TestFastIDCT() {
#if HWY_TARGET == HWY_NEON
  auto pixels_mem = hwy::AllocateAligned<float>(N * M);
  float* pixels = pixels_mem.get();
  auto dct_mem = hwy::AllocateAligned<float>(N * M);
  float* dct = dct_mem.get();
  auto dct_i_mem = hwy::AllocateAligned<int16_t>(N * M);
  int16_t* dct_i = dct_i_mem.get();
  auto dct_in_mem = hwy::AllocateAligned<int16_t>(N * M);
  int16_t* dct_in = dct_in_mem.get();
  auto idct_mem = hwy::AllocateAligned<int16_t>(N * M);
  int16_t* idct = idct_mem.get();

  auto scratch_space_mem = hwy::AllocateAligned<float>(N * M * 2);
  float* scratch_space = scratch_space_mem.get();
  auto scratch_space_i_mem = hwy::AllocateAligned<int16_t>(N * M * 2);
  int16_t* scratch_space_i = scratch_space_i_mem.get();

  Rng rng(0);
  for (size_t i = 0; i < N * M; i++) {
    pixels[i] = rng.UniformF(-1, 1);
  }
  ComputeScaledDCT<M, N>()(DCTFrom(pixels, N), dct, scratch_space);
  size_t integer_bits = std::max(FastIDCTIntegerBits(FastDCTTag<N>()),
                                 FastIDCTIntegerBits(FastDCTTag<M>()));
  // Enough range for [-2, 2] output values.
  ASSERT_LE(integer_bits, 14);
  float scale = (1 << (14 - integer_bits));
  for (size_t i = 0; i < N * M; i++) {
    dct_i[i] = std::round(dct[i] * scale);
  }

  for (size_t j = 0; j < 40000000 / (M * N); j++) {
    memcpy(dct_in, dct_i, sizeof(*dct_i) * N * M);
    ComputeFastScaledIDCT<M, N>()(dct_in, idct, N, scratch_space_i);
  }
  float max_error = 0;
  for (size_t i = 0; i < M * N; i++) {
    float err = std::abs(idct[i] * (1.0f / scale) - pixels[i]);
    if (std::abs(err) > max_error) {
      max_error = std::abs(err);
    }
  }
  EXPECT_LE(max_error, 2.0 / 256);
  printf("max error: %f mantissa bits: %d\n", max_error,
         14 - (int)integer_bits);
#endif
}

template <size_t N, size_t M>
HWY_NOINLINE void TestFloatIDCT() {
  auto pixels_mem = hwy::AllocateAligned<float>(N * M);
  float* pixels = pixels_mem.get();
  auto dct_mem = hwy::AllocateAligned<float>(N * M);
  float* dct = dct_mem.get();
  auto idct_mem = hwy::AllocateAligned<float>(N * M);
  float* idct = idct_mem.get();

  auto dct_in_mem = hwy::AllocateAligned<float>(N * M);
  float* dct_in = dct_mem.get();

  auto scratch_space_mem = hwy::AllocateAligned<float>(N * M * 2);
  float* scratch_space = scratch_space_mem.get();

  Rng rng(0);
  for (size_t i = 0; i < N * M; i++) {
    pixels[i] = rng.UniformF(-1, 1);
  }
  ComputeScaledDCT<M, N>()(DCTFrom(pixels, N), dct, scratch_space);

  for (size_t j = 0; j < 40000000 / (M * N); j++) {
    memcpy(dct_in, dct, sizeof(*dct) * N * M);
    ComputeScaledIDCT<M, N>()(dct_in, DCTTo(idct, N), scratch_space);
  }
  float max_error = 0;
  for (size_t i = 0; i < M * N; i++) {
    float err = std::abs(idct[i] - pixels[i]);
    if (std::abs(err) > max_error) {
      max_error = std::abs(err);
    }
  }
  EXPECT_LE(max_error, 1e-5);
  printf("max error: %e\n", max_error);
}

HWY_NOINLINE void TestFastTranspose8x8() { TestFastTranspose<8, 8>(); }
HWY_NOINLINE void TestFloatTranspose8x8() { TestFloatTranspose<8, 8>(); }
HWY_NOINLINE void TestFastIDCT8x8() { TestFastIDCT<8, 8>(); }
HWY_NOINLINE void TestFloatIDCT8x8() { TestFloatIDCT<8, 8>(); }
HWY_NOINLINE void TestFastTranspose8x16() { TestFastTranspose<8, 16>(); }
HWY_NOINLINE void TestFloatTranspose8x16() { TestFloatTranspose<8, 16>(); }
HWY_NOINLINE void TestFastIDCT8x16() { TestFastIDCT<8, 16>(); }
HWY_NOINLINE void TestFloatIDCT8x16() { TestFloatIDCT<8, 16>(); }
HWY_NOINLINE void TestFastTranspose8x32() { TestFastTranspose<8, 32>(); }
HWY_NOINLINE void TestFloatTranspose8x32() { TestFloatTranspose<8, 32>(); }
HWY_NOINLINE void TestFastIDCT8x32() { TestFastIDCT<8, 32>(); }
HWY_NOINLINE void TestFloatIDCT8x32() { TestFloatIDCT<8, 32>(); }
HWY_NOINLINE void TestFastTranspose16x8() { TestFastTranspose<16, 8>(); }
HWY_NOINLINE void TestFloatTranspose16x8() { TestFloatTranspose<16, 8>(); }
HWY_NOINLINE void TestFastIDCT16x8() { TestFastIDCT<16, 8>(); }
HWY_NOINLINE void TestFloatIDCT16x8() { TestFloatIDCT<16, 8>(); }
HWY_NOINLINE void TestFastTranspose16x16() { TestFastTranspose<16, 16>(); }
HWY_NOINLINE void TestFloatTranspose16x16() { TestFloatTranspose<16, 16>(); }
HWY_NOINLINE void TestFastIDCT16x16() { TestFastIDCT<16, 16>(); }
HWY_NOINLINE void TestFloatIDCT16x16() { TestFloatIDCT<16, 16>(); }
HWY_NOINLINE void TestFastTranspose16x32() { TestFastTranspose<16, 32>(); }
HWY_NOINLINE void TestFloatTranspose16x32() { TestFloatTranspose<16, 32>(); }
HWY_NOINLINE void TestFastIDCT16x32() { TestFastIDCT<16, 32>(); }
HWY_NOINLINE void TestFloatIDCT16x32() { TestFloatIDCT<16, 32>(); }
HWY_NOINLINE void TestFastTranspose32x8() { TestFastTranspose<32, 8>(); }
HWY_NOINLINE void TestFloatTranspose32x8() { TestFloatTranspose<32, 8>(); }
HWY_NOINLINE void TestFastIDCT32x8() { TestFastIDCT<32, 8>(); }
HWY_NOINLINE void TestFloatIDCT32x8() { TestFloatIDCT<32, 8>(); }
HWY_NOINLINE void TestFastTranspose32x16() { TestFastTranspose<32, 16>(); }
HWY_NOINLINE void TestFloatTranspose32x16() { TestFloatTranspose<32, 16>(); }
HWY_NOINLINE void TestFastIDCT32x16() { TestFastIDCT<32, 16>(); }
HWY_NOINLINE void TestFloatIDCT32x16() { TestFloatIDCT<32, 16>(); }
HWY_NOINLINE void TestFastTranspose32x32() { TestFastTranspose<32, 32>(); }
HWY_NOINLINE void TestFloatTranspose32x32() { TestFloatTranspose<32, 32>(); }
HWY_NOINLINE void TestFastIDCT32x32() { TestFastIDCT<32, 32>(); }
HWY_NOINLINE void TestFloatIDCT32x32() { TestFloatIDCT<32, 32>(); }
HWY_NOINLINE void TestFastTranspose32x64() { TestFastTranspose<32, 64>(); }
HWY_NOINLINE void TestFloatTranspose32x64() { TestFloatTranspose<32, 64>(); }
HWY_NOINLINE void TestFastIDCT32x64() { TestFastIDCT<32, 64>(); }
HWY_NOINLINE void TestFloatIDCT32x64() { TestFloatIDCT<32, 64>(); }
HWY_NOINLINE void TestFastTranspose64x32() { TestFastTranspose<64, 32>(); }
HWY_NOINLINE void TestFloatTranspose64x32() { TestFloatTranspose<64, 32>(); }
HWY_NOINLINE void TestFastIDCT64x32() { TestFastIDCT<64, 32>(); }
HWY_NOINLINE void TestFloatIDCT64x32() { TestFloatIDCT<64, 32>(); }
HWY_NOINLINE void TestFastTranspose64x64() { TestFastTranspose<64, 64>(); }
HWY_NOINLINE void TestFloatTranspose64x64() { TestFloatTranspose<64, 64>(); }
HWY_NOINLINE void TestFastIDCT64x64() { TestFastIDCT<64, 64>(); }
HWY_NOINLINE void TestFloatIDCT64x64() { TestFloatIDCT<64, 64>(); }
HWY_NOINLINE void TestFastTranspose64x128() { TestFastTranspose<64, 128>(); }
HWY_NOINLINE void TestFloatTranspose64x128() { TestFloatTranspose<64, 128>(); }
HWY_NOINLINE void TestFastIDCT64x128() { TestFastIDCT<64, 128>(); }
HWY_NOINLINE void TestFloatIDCT64x128() { TestFloatIDCT<64, 128>(); }
HWY_NOINLINE void TestFastTranspose128x64() { TestFastTranspose<128, 64>(); }
HWY_NOINLINE void TestFloatTranspose128x64() { TestFloatTranspose<128, 64>(); }
HWY_NOINLINE void TestFastIDCT128x64() { TestFastIDCT<128, 64>(); }
HWY_NOINLINE void TestFloatIDCT128x64() { TestFloatIDCT<128, 64>(); }
HWY_NOINLINE void TestFastTranspose128x128() { TestFastTranspose<128, 128>(); }
HWY_NOINLINE void TestFloatTranspose128x128() {
  TestFloatTranspose<128, 128>();
}
HWY_NOINLINE void TestFastIDCT128x128() { TestFastIDCT<128, 128>(); }
HWY_NOINLINE void TestFloatIDCT128x128() { TestFloatIDCT<128, 128>(); }
HWY_NOINLINE void TestFastTranspose128x256() { TestFastTranspose<128, 256>(); }
HWY_NOINLINE void TestFloatTranspose128x256() {
  TestFloatTranspose<128, 256>();
}
HWY_NOINLINE void TestFastIDCT128x256() { TestFastIDCT<128, 256>(); }
HWY_NOINLINE void TestFloatIDCT128x256() { TestFloatIDCT<128, 256>(); }
HWY_NOINLINE void TestFastTranspose256x128() { TestFastTranspose<256, 128>(); }
HWY_NOINLINE void TestFloatTranspose256x128() {
  TestFloatTranspose<256, 128>();
}
HWY_NOINLINE void TestFastIDCT256x128() { TestFastIDCT<256, 128>(); }
HWY_NOINLINE void TestFloatIDCT256x128() { TestFloatIDCT<256, 128>(); }
HWY_NOINLINE void TestFastTranspose256x256() { TestFastTranspose<256, 256>(); }
HWY_NOINLINE void TestFloatTranspose256x256() {
  TestFloatTranspose<256, 256>();
}
HWY_NOINLINE void TestFastIDCT256x256() { TestFastIDCT<256, 256>(); }
HWY_NOINLINE void TestFloatIDCT256x256() { TestFloatIDCT<256, 256>(); }

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

class FastDCTTargetTest : public hwy::TestWithParamTarget {};
HWY_TARGET_INSTANTIATE_TEST_SUITE_P(FastDCTTargetTest);

HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatTranspose8x8);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastTranspose8x8);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatTranspose8x16);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastTranspose8x16);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatTranspose8x32);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastTranspose8x32);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatTranspose16x8);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastTranspose16x8);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatTranspose16x16);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastTranspose16x16);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatTranspose16x32);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastTranspose16x32);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatTranspose32x8);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastTranspose32x8);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatTranspose32x16);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastTranspose32x16);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatTranspose32x32);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastTranspose32x32);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatTranspose32x64);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastTranspose32x64);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatTranspose64x32);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastTranspose64x32);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatTranspose64x64);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastTranspose64x64);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatTranspose64x128);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastTranspose64x128);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatTranspose128x64);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastTranspose128x64);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatTranspose128x128);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastTranspose128x128);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatTranspose128x256);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastTranspose128x256);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatTranspose256x128);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastTranspose256x128);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatTranspose256x256);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastTranspose256x256);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatIDCT8x8);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastIDCT8x8);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatIDCT8x16);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastIDCT8x16);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatIDCT8x32);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastIDCT8x32);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatIDCT16x8);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastIDCT16x8);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatIDCT16x16);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastIDCT16x16);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatIDCT16x32);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastIDCT16x32);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatIDCT32x8);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastIDCT32x8);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatIDCT32x16);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastIDCT32x16);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatIDCT32x32);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastIDCT32x32);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatIDCT32x64);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastIDCT32x64);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatIDCT64x32);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastIDCT64x32);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatIDCT64x64);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastIDCT64x64);
/*
 * DCT-128 and above have very large errors just by rounding inputs.
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatIDCT64x128);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastIDCT64x128);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatIDCT128x64);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastIDCT128x64);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatIDCT128x128);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastIDCT128x128);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatIDCT128x256);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastIDCT128x256);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatIDCT256x128);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastIDCT256x128);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFloatIDCT256x256);
HWY_EXPORT_AND_TEST_P(FastDCTTargetTest, TestFastIDCT256x256);
*/

}  // namespace jxl
#endif  // HWY_ONCE
