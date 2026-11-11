// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_rdoq.h"

#include <jxl/cms.h>
#include <jxl/encode.h>
#include <jxl/memory_manager.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "lib/extras/codec_in_out.h"
#include "lib/extras/enc/jxl.h"
#include "lib/jxl/ac_context.h"
#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/common.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/dec_cache.h"
#include "lib/jxl/dec_frame.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/headers.h"
#include "lib/jxl/icc_codec.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/pack_signed.h"
#include "lib/jxl/padded_bytes.h"
#include "lib/jxl/quant_weights.h"
#include "lib/jxl/quantizer.h"
#include "lib/jxl/test_image.h"
#include "lib/jxl/test_memory_manager.h"
#include "lib/jxl/test_utils.h"
#include "lib/jxl/testing.h"

namespace jxl {
namespace {

float DebiasForTest(size_t c, int32_t k) {
  if (k == 0) return 0.0f;
  if (k == 1) return kDefaultQuantBias[c];
  if (k == -1) return -kDefaultQuantBias[c];
  float fk = static_cast<float>(k);
  return fk - kDefaultQuantBias[3] / fk;
}

struct DecodedRDOInspection {
  std::unique_ptr<CodecInOut> io;
  std::unique_ptr<PassesDecoderState> dec_state;
  FrameHeader last_frame_header;

  explicit DecodedRDOInspection(const CodecMetadata& metadata)
      : last_frame_header(&metadata) {}
};

void DecodeCompressedInspect(const std::vector<uint8_t>& compressed,
                             std::unique_ptr<CodecMetadata>* metadata_out,
                             std::unique_ptr<DecodedRDOInspection>* out) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  *metadata_out = jxl::make_unique<CodecMetadata>();
  CodecMetadata* metadata = metadata_out->get();
  *out = jxl::make_unique<DecodedRDOInspection>(*metadata);
  (*out)->io = jxl::make_unique<CodecInOut>(memory_manager);

  Bytes compressed_bytes(compressed);
  BitReader reader(compressed_bytes);
  ASSERT_EQ(reader.ReadFixedBits<16>(), 0x0AFFu);
  ASSERT_TRUE(ReadSizeHeader(&reader, &metadata->size));
  ASSERT_TRUE(ReadImageMetadata(&reader, &metadata->m));
  metadata->transform_data.nonserialized_xyb_encoded = metadata->m.xyb_encoded;
  ASSERT_TRUE(Bundle::Read(&reader, &metadata->transform_data));
  if (metadata->m.color_encoding.WantICC()) {
    std::vector<uint8_t> icc;
    ASSERT_TRUE(test::ReadICC(&reader, &icc));
    ASSERT_TRUE(
        metadata->m.color_encoding.SetICC(std::move(icc), JxlGetDefaultCms()));
  }
  (*out)->io->metadata = *metadata;
  (*out)->dec_state = jxl::make_unique<PassesDecoderState>(memory_manager);
  ASSERT_TRUE(
      (*out)->dec_state->output_encoding_info.SetFromMetadata(*metadata));
  ASSERT_TRUE(reader.JumpToByteBoundary());

  do {
    (*out)->io->frames.emplace_back(memory_manager, &metadata->m);
    size_t frame_start = reader.TotalBitsConsumed() / kBitsPerByte;
    size_t size_left = compressed.size() - frame_start;
    ASSERT_TRUE(DecodeFrame((*out)->dec_state.get(), /*pool=*/nullptr,
                            compressed.data() + frame_start, size_left,
                            &(*out)->last_frame_header,
                            &(*out)->io->frames.back(), *metadata));
    reader.SkipBits((*out)->io->frames.back().decoded_bytes() * kBitsPerByte);
  } while ((*out)->last_frame_header.frame_type != FrameType::kRegularFrame &&
           (*out)->last_frame_header.frame_type != FrameType::kSkipProgressive);
  ASSERT_TRUE(reader.Close());
}

std::unique_ptr<CodecInOut> MakeSRGBCodecInOut(const Image3F& srgb01) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  auto io = jxl::make_unique<CodecInOut>(memory_manager);
  io->metadata.m.SetUintSamples(8);
  io->metadata.m.color_encoding = ColorEncoding::SRGB();
  StatusOr<Image3F> copy_or =
      Image3F::Create(memory_manager, srgb01.xsize(), srgb01.ysize());
  Check(copy_or.ok());
  Image3F copy = std::move(copy_or).value_();
  Check(CopyImageTo(Rect(srgb01), srgb01, Rect(copy), &copy));
  Check(io->SetFromImage(std::move(copy), io->metadata.m.color_encoding));
  return io;
}

TEST(EncRDOQTest, TrellisMatchesExhaustiveBruteForceOnSparse8x8) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  DequantMatrices dequant;
  ASSERT_TRUE(dequant.EnsureComputed(
      memory_manager, 1u << static_cast<uint32_t>(AcStrategyType::DCT)));
  Quantizer quantizer(dequant);
  JXL_TEST_ASSIGN_OR_DIE(ImageI rqf, ImageI::Create(memory_manager, 1, 1));
  quantizer.SetQuant(0.25f, 0.25f, &rqf);
  const int32_t qf = rqf.Row(0)[0];
  const float q_target = quantizer.Scale() * static_cast<float>(qf);

  BlockCtxMap bcm;
  const size_t channel = 1;
  const size_t block_ctx = bcm.Context(/*dc_idx=*/0, qf, /*ord=*/0, channel);
  const int32_t predicted_nzeros = 4;

  const AcStrategy acs = AcStrategy::FromRawStrategy(AcStrategyType::DCT);
  coeff_order_t order[64];
  acs.ComputeNaturalCoeffOrder(order);

  RDORateModel rate_model;
  rate_model.InitPreset(bcm, /*butteraugli_distance=*/1.0f,
                        /*decoding_speed_tier=*/0);

  const float* inv = quantizer.InvDequantMatrix(AcStrategyType::DCT, channel);

  // 7 active AC positions with mixed positive and negative borderline
  // magnitudes.
  const std::array<size_t, 7> active_p = {1, 3, 6, 11, 18, 29, 42};
  const std::array<float, 7> target_u = {1.75f, -0.78f, 1.32f, -0.58f,
                                         0.92f, 1.62f,  -0.66f};

  float coeffs_in[64] = {};
  for (size_t i = 0; i < active_p.size(); ++i) {
    const coeff_order_t idx = order[active_p[i]];
    ASSERT_GT(inv[idx], 0.0f);
    coeffs_in[idx] = target_u[i] / (inv[idx] * q_target);
  }

  std::array<float, 64> u_order = {};
  for (size_t p = 1; p < 64; ++p) {
    u_order[p] = coeffs_in[order[p]] * (inv[order[p]] * q_target);
  }

  const float s_base = q_target / (quantizer.Scale() * static_cast<float>(qf));
  const uint32_t nz_ctx =
      bcm.NonZeroContext(static_cast<uint32_t>(predicted_nzeros),
                         static_cast<uint32_t>(block_ctx));
  const size_t zd_off =
      bcm.ZeroDensityContextsOffset(static_cast<uint32_t>(block_ctx));

  for (float lambda : {0.02f, 0.06f, 0.18f}) {
    int32_t quantized_dp[64] = {};
    RDOQTrellisResult dp_res = RunTrellisRDOQSingleBlockForTest(
        channel, AcStrategyType::DCT, coeffs_in, quantizer,
        /*qm_multiplier=*/1.0f, qf, q_target, predicted_nzeros, block_ctx, bcm,
        order, rate_model, lambda, /*band_width=*/64, /*multi_level=*/2,
        quantized_dp);

    // Exhaustively enumerate all 5^7 = 78,125 assignments in {-2, -1, 0, 1, 2}.
    float best_brute_j = std::numeric_limits<float>::infinity();
    std::array<int32_t, 64> k_assign = {};
    constexpr int32_t kTotalCombos = 5 * 5 * 5 * 5 * 5 * 5 * 5;
    for (int32_t mask = 0; mask < kTotalCombos; ++mask) {
      int32_t tmp = mask;
      int32_t nz = 0;
      for (size_t i = 0; i < active_p.size(); ++i) {
        int32_t kv = (tmp % 5) - 2;
        tmp /= 5;
        k_assign[active_p[i]] = kv;
        if (kv != 0) ++nz;
      }

      float dist = 0.0f;
      for (size_t p = 1; p < 64; ++p) {
        const float r =
            u_order[p] - s_base * DebiasForTest(channel, k_assign[p]);
        dist += r * r;
      }

      float rate = rate_model.TokenCost(nz_ctx, static_cast<uint32_t>(nz));
      int32_t rem = nz;
      size_t prev = (static_cast<size_t>(nz) > 64 / 16) ? 0 : 1;
      for (size_t p = 1; p < 64 && rem > 0; ++p) {
        const size_t ctx = zd_off + ZeroDensityContext(static_cast<size_t>(rem),
                                                       p, 1, 0, prev);
        const int32_t kv = k_assign[p];
        rate += rate_model.TokenCost(ctx, PackSigned(kv));
        prev = (kv != 0) ? 1 : 0;
        rem -= static_cast<int32_t>(prev);
      }

      const float j = dist + lambda * rate;
      if (j < best_brute_j) best_brute_j = j;
    }

    EXPECT_NEAR(dp_res.rd_cost, best_brute_j, 1e-4f) << "lambda=" << lambda;
  }
}

TEST(EncRDOQTest, LambdaAndTrellisMonotonicity) {
  const std::array<float, 5> distances = {0.5f, 1.0f, 2.0f, 4.0f, 8.0f};
  for (size_t i = 1; i < distances.size(); ++i) {
    EXPECT_LT(ComputeRDOLambda(distances[i]),
              ComputeRDOLambda(distances[i - 1]));
  }

  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  DequantMatrices dequant;
  ASSERT_TRUE(dequant.EnsureComputed(
      memory_manager, 1u << static_cast<uint32_t>(AcStrategyType::DCT)));
  Quantizer quantizer(dequant);
  JXL_TEST_ASSIGN_OR_DIE(ImageI rqf, ImageI::Create(memory_manager, 1, 1));
  quantizer.SetQuant(0.25f, 0.25f, &rqf);
  const int32_t qf = rqf.Row(0)[0];
  const float q_target = quantizer.Scale() * static_cast<float>(qf);

  BlockCtxMap bcm;
  const size_t channel = 1;
  const size_t block_ctx = bcm.Context(0, qf, 0, channel);
  const AcStrategy acs = AcStrategy::FromRawStrategy(AcStrategyType::DCT);
  coeff_order_t order[64];
  acs.ComputeNaturalCoeffOrder(order);

  RDORateModel rate_model;
  rate_model.InitPreset(bcm, 1.0f, 0);

  const float* inv = quantizer.InvDequantMatrix(AcStrategyType::DCT, channel);
  float coeffs_in[64] = {};
  for (size_t p = 1; p < 64; ++p) {
    const coeff_order_t idx = order[p];
    if (inv[idx] <= 0.0f) continue;
    const float mag = 2.4f / (1.0f + 0.08f * static_cast<float>(p));
    const float sign = (p % 3 == 0) ? -1.0f : 1.0f;
    coeffs_in[idx] = sign * mag / (inv[idx] * q_target);
  }

  float prev_rate = std::numeric_limits<float>::infinity();
  int32_t prev_nz = std::numeric_limits<int32_t>::max();
  float prev_dist = -1.0f;
  for (float lambda : {0.01f, 0.04f, 0.12f, 0.35f}) {
    int32_t q_out[64] = {};
    RDOQTrellisResult res = RunTrellisRDOQSingleBlockForTest(
        channel, AcStrategyType::DCT, coeffs_in, quantizer, 1.0f, qf, q_target,
        /*predicted_nzeros=*/16, block_ctx, bcm, order, rate_model, lambda,
        /*band_width=*/24, /*multi_level=*/2, q_out);
    EXPECT_LE(res.rate_bits, prev_rate + 1e-4f);
    EXPECT_LE(res.nzeros, prev_nz);
    EXPECT_GE(res.dist, prev_dist - 1e-4f);
    prev_rate = res.rate_bits;
    prev_nz = res.nzeros;
    prev_dist = res.dist;
  }
}

TEST(EncRDOQTest, DeadzoneWideningAQSmoothnessAndValidity) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  constexpr size_t kW = 128;
  constexpr size_t kH = 64;
  JXL_TEST_ASSIGN_OR_DIE(Image3F img, Image3F::Create(memory_manager, kW, kH));
  for (size_t y = 0; y < kH; ++y) {
    for (size_t x = 0; x < kW; ++x) {
      const float base =
          0.45f + 0.04f * std::sin(0.05f * static_cast<float>(x));
      const float rip = 0.02f * std::sin(0.18f * static_cast<float>(x)) *
                        std::cos(0.15f * static_cast<float>(y));
      img.PlaneRow(0, y)[x] = Clamp1(base + rip, 0.0f, 1.0f);
      img.PlaneRow(1, y)[x] = Clamp1(base + 0.5f * rip, 0.0f, 1.0f);
      img.PlaneRow(2, y)[x] = Clamp1(base - 0.5f * rip, 0.0f, 1.0f);
    }
  }
  auto io = MakeSRGBCodecInOut(img);

  CompressParams cparams;
  cparams.speed_tier = SpeedTier::kSquirrel;
  cparams.butteraugli_distance = 1.5f;
  std::vector<uint8_t> compressed;
  ASSERT_TRUE(test::EncodeFile(cparams, io.get(), &compressed));

  std::unique_ptr<CodecMetadata> metadata;
  std::unique_ptr<DecodedRDOInspection> dec;
  DecodeCompressedInspect(compressed, &metadata, &dec);

  // Note: ModularFrameDecoder only writes decoded raw_quant_field entries at
  // AcStrategy::IsFirstBlock() block origins.
  const PassesSharedState& shared = *dec->dec_state->shared;
  const size_t bx_cnt = shared.frame_dim.xsize_blocks;
  const size_t by_cnt = shared.frame_dim.ysize_blocks;
  std::vector<int32_t> resolved_qf(bx_cnt * by_cnt, 0);
  for (size_t by = 0; by < by_cnt; ++by) {
    const int32_t* qrow = shared.raw_quant_field.ConstRow(by);
    AcStrategyRow acs_row = shared.ac_strategy.ConstRow(by);
    for (size_t bx = 0; bx < bx_cnt; ++bx) {
      const AcStrategy acs = acs_row[bx];
      if (acs.IsFirstBlock()) {
        const int32_t ref_qf = qrow[bx];
        EXPECT_GE(ref_qf, 1);
        EXPECT_LE(ref_qf, Quantizer::kQuantMax);
        for (size_t iy = 0; iy < acs.covered_blocks_y(); ++iy) {
          for (size_t ix = 0; ix < acs.covered_blocks_x(); ++ix) {
            const size_t idx = (by + iy) * bx_cnt + (bx + ix);
            EXPECT_EQ(resolved_qf[idx], 0);
            resolved_qf[idx] = ref_qf;
          }
        }
      }
    }
  }

  for (size_t idx = 0; idx < resolved_qf.size(); ++idx) {
    EXPECT_GE(resolved_qf[idx], 1);
    EXPECT_LE(resolved_qf[idx], Quantizer::kQuantMax);
  }

  size_t equal_pairs = 0;
  size_t total_pairs = 0;
  for (size_t by = 0; by < by_cnt; ++by) {
    for (size_t bx = 1; bx < bx_cnt; ++bx) {
      ++total_pairs;
      if (resolved_qf[by * bx_cnt + bx] ==
          resolved_qf[by * bx_cnt + (bx - 1)]) {
        ++equal_pairs;
      }
    }
  }
  ASSERT_GT(total_pairs, 0u);
  EXPECT_GE(equal_pairs * 2, total_pairs);
}

TEST(EncRDOQTest, GuillotineDPSelectsMixedAndPartialTiles) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  constexpr size_t kSize = 64;
  JXL_TEST_ASSIGN_OR_DIE(Image3F mixed,
                         Image3F::Create(memory_manager, kSize, kSize));
  for (size_t y = 0; y < kSize; ++y) {
    for (size_t x = 0; x < kSize; ++x) {
      if (x < 32) {
        const float v =
            0.25f + 0.50f * (static_cast<float>(x + y) / (32.0f + kSize));
        mixed.PlaneRow(0, y)[x] = v;
        mixed.PlaneRow(1, y)[x] = v * 0.95f;
        mixed.PlaneRow(2, y)[x] = v * 0.90f;
      } else {
        // Non-periodic isolated sharp micro-steps shifted uniquely inside
        // each 8x8 block so large merged blocks would ring into flat areas.
        const size_t bx = (x - 32) / 8;
        const size_t by = y / 8;
        const size_t lx = x % 8;
        const size_t ly = y % 8;
        const size_t sx = (bx * 3 + by * 5 + 1) % 6 + 1;
        const size_t sy = (bx * 5 + by * 2 + 2) % 6 + 1;
        const bool step = (lx == sx || lx == sx + 1) && (ly == sy);
        mixed.PlaneRow(0, y)[x] = step ? 0.98f : 0.45f;
        mixed.PlaneRow(1, y)[x] = step ? 0.08f : 0.45f;
        mixed.PlaneRow(2, y)[x] = step ? 0.92f : 0.45f;
      }
    }
  }
  auto io_mixed = MakeSRGBCodecInOut(mixed);

  for (SpeedTier tier :
       {SpeedTier::kSquirrel, SpeedTier::kKitten, SpeedTier::kTortoise}) {
    CompressParams cparams;
    cparams.speed_tier = tier;
    cparams.butteraugli_distance = 1.0f;
    std::vector<uint8_t> compressed;
    ASSERT_TRUE(test::EncodeFile(cparams, io_mixed.get(), &compressed));

    std::unique_ptr<CodecMetadata> metadata;
    std::unique_ptr<DecodedRDOInspection> dec;
    DecodeCompressedInspect(compressed, &metadata, &dec);

    const AcStrategyImage& acs_img = dec->dec_state->shared->ac_strategy;
    bool left_has_large = false;
    bool right_has_small = false;
    for (size_t by = 0; by < 8; ++by) {
      AcStrategyRow row = acs_img.ConstRow(by);
      for (size_t bx = 0; bx < 8; ++bx) {
        const AcStrategy acs = row[bx];
        const size_t cov = acs.covered_blocks_x() * acs.covered_blocks_y();
        if (bx < 4 && cov >= 4) {
          left_has_large = true;
        }
        if (bx >= 4 && cov == 1) {
          right_has_small = true;
        }
      }
    }
    EXPECT_TRUE(left_has_large) << "tier=" << static_cast<int>(tier);
    EXPECT_TRUE(right_has_small) << "tier=" << static_cast<int>(tier);
  }

  for (std::pair<size_t, size_t> dims :
       {std::pair<size_t, size_t>{40, 40}, std::pair<size_t, size_t>{72, 56}}) {
    JXL_TEST_ASSIGN_OR_DIE(
        Image3F img, Image3F::Create(memory_manager, dims.first, dims.second));
    for (size_t y = 0; y < dims.second; ++y) {
      for (size_t x = 0; x < dims.first; ++x) {
        img.PlaneRow(0, y)[x] =
            0.2f + 0.6f * std::sin(0.25f * static_cast<float>(x)) *
                       std::cos(0.18f * static_cast<float>(y));
        img.PlaneRow(1, y)[x] =
            0.5f * static_cast<float>(x) / static_cast<float>(dims.first) +
            0.3f;
        img.PlaneRow(2, y)[x] =
            0.5f * static_cast<float>(y) / static_cast<float>(dims.second) +
            0.2f;
      }
    }
    auto io_part = MakeSRGBCodecInOut(img);
    for (SpeedTier tier :
         {SpeedTier::kSquirrel, SpeedTier::kKitten, SpeedTier::kTortoise}) {
      CompressParams cparams;
      cparams.speed_tier = tier;
      cparams.butteraugli_distance = 1.5f;
      auto io2 = jxl::make_unique<CodecInOut>(memory_manager);
      std::stringstream failures;
      extras::JXLDecompressParams dparams;
      EXPECT_TRUE(
          test::Roundtrip(io_part.get(), cparams, dparams, io2.get(), failures))
          << failures.str();
      const float d =
          test::ButteraugliDistance(io_part->frames, io2->frames,
                                    ButteraugliParams(), *JxlGetDefaultCms());
      EXPECT_LT(d, 4.0f) << "dims=" << dims.first << "x" << dims.second
                         << " tier=" << static_cast<int>(tier);
    }
  }
}

TEST(EncRDOQTest, VarDCTFeaturesDefaultOffAndExplicitOverrideOn) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  constexpr size_t kDim = 256;
  JXL_TEST_ASSIGN_OR_DIE(Image3F pat,
                         Image3F::Create(memory_manager, kDim, kDim));
  ZeroFillImage(&pat);
  for (size_t y = 0; y < kDim; ++y) {
    float* row1 = pat.PlaneRow(1, y);
    for (size_t x = 0; x < kDim; ++x) {
      if (x % 4 == 0 && (y / 32) % 4 == 0) {
        row1[x] = 127.0f / 255.0f;
      }
    }
  }
  auto io = MakeSRGBCodecInOut(pat);

  // Default VarDCT at e7 d=4.0: patches, splines, and noise must remain off.
  {
    CompressParams cparams;
    cparams.speed_tier = SpeedTier::kSquirrel;
    cparams.butteraugli_distance = 4.0f;
    std::vector<uint8_t> compressed;
    ASSERT_TRUE(test::EncodeFile(cparams, io.get(), &compressed));
    std::unique_ptr<CodecMetadata> metadata;
    std::unique_ptr<DecodedRDOInspection> dec;
    DecodeCompressedInspect(compressed, &metadata, &dec);
    EXPECT_EQ(dec->last_frame_header.flags & FrameHeader::kPatches, 0u);
    EXPECT_EQ(dec->last_frame_header.flags & FrameHeader::kSplines, 0u);
    EXPECT_EQ(dec->last_frame_header.flags & FrameHeader::kNoise, 0u);
    EXPECT_FALSE(dec->dec_state->shared->image_features.patches.HasAny());
  }

  // Explicit Override::kOn for patches must re-enable patches.
  {
    CompressParams cparams;
    cparams.speed_tier = SpeedTier::kSquirrel;
    cparams.butteraugli_distance = 0.5f;
    cparams.patches = Override::kOn;
    std::vector<uint8_t> compressed;
    ASSERT_TRUE(test::EncodeFile(cparams, io.get(), &compressed));
    std::unique_ptr<CodecMetadata> metadata;
    std::unique_ptr<DecodedRDOInspection> dec;
    DecodeCompressedInspect(compressed, &metadata, &dec);
    EXPECT_NE(dec->last_frame_header.flags & FrameHeader::kPatches, 0u);
    EXPECT_TRUE(dec->dec_state->shared->image_features.patches.HasAny());
  }

  // Noise synthesis: off by default in VarDCT on a photographic noisy image,
  // enabled when JXL_ENC_FRAME_SETTING_NOISE = 1 is explicitly passed.
  const std::vector<uint8_t> orig =
      test::ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  test::TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();
  {
    extras::JXLCompressParams jparams;
    jparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 7);
    jparams.distance = 2.0f;
    std::vector<uint8_t> compressed;
    ASSERT_TRUE(extras::EncodeImageJXL(jparams, t.ppf(), nullptr, &compressed));
    std::unique_ptr<CodecMetadata> metadata;
    std::unique_ptr<DecodedRDOInspection> dec;
    DecodeCompressedInspect(compressed, &metadata, &dec);
    EXPECT_EQ(dec->last_frame_header.flags & FrameHeader::kNoise, 0u);
    EXPECT_EQ(dec->last_frame_header.flags & FrameHeader::kPatches, 0u);
    EXPECT_EQ(dec->last_frame_header.flags & FrameHeader::kSplines, 0u);
  }
  {
    extras::JXLCompressParams jparams;
    jparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 7);
    jparams.AddOption(JXL_ENC_FRAME_SETTING_NOISE, 1);
    jparams.distance = 1.0f;
    std::vector<uint8_t> compressed;
    ASSERT_TRUE(extras::EncodeImageJXL(jparams, t.ppf(), nullptr, &compressed));
    std::unique_ptr<CodecMetadata> metadata;
    std::unique_ptr<DecodedRDOInspection> dec;
    DecodeCompressedInspect(compressed, &metadata, &dec);
    EXPECT_NE(dec->last_frame_header.flags & FrameHeader::kNoise, 0u);
  }
}

}  // namespace
}  // namespace jxl
