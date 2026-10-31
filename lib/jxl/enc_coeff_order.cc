// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_coeff_order.h"

#include <jxl/memory_manager.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "lib/jxl/ac_context.h"
#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/coeff_order.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/common.h"
#include "lib/jxl/dct_util.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_ans_params.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/lehmer_code.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/pack_signed.h"

namespace jxl {

struct AuxOut;
enum class LayerType : uint8_t;

std::pair<uint32_t, uint32_t> ComputeUsedOrders(
    const SpeedTier speed, const AcStrategyImage& ac_strategy,
    const Rect& rect, const bool allow_large_orders) {
  // No coefficient reordering in Falcon or faster.
  // Only uses DCT8 = 0, so bitfield = 1.
  if (speed >= SpeedTier::kFalcon) return {1, 1};

  uint32_t ret = 0;
  uint32_t ret_customize = 0;
  const int max_ord = allow_large_orders ? 8 : 6;
  size_t ord_counts[9] = {};
  size_t xsize_blocks = rect.xsize();
  size_t ysize_blocks = rect.ysize();
  // TODO(veluca): precompute when doing DCT.
  for (size_t by = 0; by < ysize_blocks; ++by) {
    AcStrategyRow acs_row = ac_strategy.ConstRow(rect, by);
    for (size_t bx = 0; bx < xsize_blocks; ++bx) {
      AcStrategy acs = acs_row[bx];
      int ord = kStrategyOrder[acs.RawStrategy()];
      ret |= 1u << ord;
      if (ord > max_ord) {
        continue;
      }
      if (acs.IsFirstBlock()) {
        ++ord_counts[ord];
      }
    }
  }
  for (int ord = 0; ord <= max_ord; ++ord) {
    if (allow_large_orders) {
      const size_t min_cnt = (ord >= 3 || xsize_blocks * ysize_blocks <= 64)
                                 ? 1
                                 : 2;
      if (ord_counts[ord] >= min_cnt) ret_customize |= 1u << ord;
    } else if (xsize_blocks * ysize_blocks <= 64) {
      if (ord_counts[ord] >= 1) ret_customize |= 1u << ord;
    } else if (ord_counts[ord] >= 4) {
      ret_customize |= 1u << ord;
    }
  }
  if (!allow_large_orders && ac_strategy.xsize() < 5 &&
      ac_strategy.ysize() < 5) {
    return {ret, 0};
  }
  if (xsize_blocks * ysize_blocks == 0) return {ret, 0};
  return {ret, ret_customize};
}

namespace {

Status TokenizePermutation(const coeff_order_t* JXL_RESTRICT order, size_t skip,
                           size_t size, std::vector<Token>* tokens) {
  std::vector<LehmerT> lehmer(size);
  std::vector<uint32_t> temp(size + 1);
  JXL_RETURN_IF_ERROR(
      ComputeLehmerCode(order, temp.data(), size, lehmer.data()));
  size_t end = size;
  while (end > skip && lehmer[end - 1] == 0) {
    --end;
  }
  tokens->emplace_back(CoeffOrderContext(size),
                       static_cast<uint32_t>(end - skip));
  uint32_t last = 0;
  for (size_t i = skip; i < end; ++i) {
    tokens->emplace_back(CoeffOrderContext(last), lehmer[i]);
    last = lehmer[i];
  }
  return true;
}

Status EncodeCoeffOrder(const coeff_order_t* JXL_RESTRICT order, AcStrategy acs,
                        std::vector<Token>* tokens, coeff_order_t* order_zigzag,
                        std::vector<coeff_order_t>& natural_order_lut) {
  const size_t llf = acs.covered_blocks_x() * acs.covered_blocks_y();
  const size_t size = kDCTBlockSize * llf;
  for (size_t i = 0; i < size; ++i) {
    order_zigzag[i] = natural_order_lut[order[i]];
  }
  JXL_RETURN_IF_ERROR(TokenizePermutation(order_zigzag, llf, size, tokens));
  return true;
}

Status EstimateOrderPayloadBits(JxlMemoryManager* memory_manager,
                                uint16_t used_orders,
                                const coeff_order_t* JXL_RESTRICT order,
                                size_t* out_bits) {
  BitWriter bw{memory_manager};
  size_t order_bits = 0;
  JXL_RETURN_IF_ERROR(
      U32Coder::CanEncode(kOrderEnc, used_orders, &order_bits));
  JXL_RETURN_IF_ERROR(
      bw.WithMaxBits(order_bits, LayerType::Order, nullptr, [&] {
        return U32Coder::Write(kOrderEnc, used_orders, &bw);
      }));
  JXL_RETURN_IF_ERROR(
      EncodeCoeffOrders(used_orders, order, &bw, LayerType::Order, nullptr));
  *out_bits = bw.BitsWritten();
  return true;
}

}  // namespace

Status ComputeCoeffOrder(SpeedTier speed, const ACImage& ac_image,
                         const AcStrategyImage& ac_strategy,
                         const FrameDimensions& frame_dim,
                         uint32_t& all_used_orders, uint32_t prev_used_acs,
                         uint32_t current_used_acs,
                         uint32_t current_used_orders,
                         coeff_order_t* JXL_RESTRICT order,
                         const ImageI* raw_quant_field, bool exact_rdo) {
  JxlMemoryManager* memory_manager = ac_strategy.memory_manager();
  const bool run_exact_rdo = exact_rdo && ac_image.Type() == ACType::k32 &&
                             prev_used_acs == 0 && all_used_orders == 0;
  if (run_exact_rdo) {
    struct PosAndCount {
      uint32_t pos;
      uint64_t count_and_idx;
    };
    size_t mem_bytes = AcStrategy::kMaxCoeffArea * sizeof(PosAndCount);
    JXL_ASSIGN_OR_RETURN(auto mem,
                         AlignedMemory::Create(memory_manager, mem_bytes));
    PosAndCount* pos_and_val = mem.address<PosAndCount>();

    // Initialize natural orders for all used transforms and kOrderEnc shortcut
    // slots (0x13 and 0x5F) so padded shortcut masks are always valid.
    const uint32_t init_mask = current_used_acs | 0x5Fu;
    std::vector<coeff_order_t> natural_order_buffer;
    uint8_t rep_strategy_for_ord[kNumOrders] = {};
    {
      uint16_t init_done = 0;
      for (uint8_t o = 0; o < AcStrategy::kNumValidStrategies; ++o) {
        const uint8_t ord = kStrategyOrder[o];
        if (init_done & (1u << ord)) continue;
        init_done |= (1u << ord);
        rep_strategy_for_ord[ord] = o;
        if (((init_mask >> ord) & 1u) == 0) continue;
        AcStrategy acs = AcStrategy::FromRawStrategy(o);
        const size_t sz =
            kDCTBlockSize * acs.covered_blocks_x() * acs.covered_blocks_y();
        if (natural_order_buffer.size() < sz) natural_order_buffer.resize(sz);
        acs.ComputeNaturalCoeffOrder(natural_order_buffer.data());
        for (size_t c = 0; c < 3; ++c) {
          const size_t offset = CoeffOrderOffset(ord, c);
          memcpy(&order[offset], natural_order_buffer.data(),
                 sz * sizeof(*order));
        }
      }
    }
    if (current_used_orders == 0) {
      return true;
    }

    struct BlockRef {
      const int32_t* ptr[3];
      uint16_t nz[3];
      uint32_t qf;
    };
    std::vector<BlockRef> blocks_by_ord[9];
    std::vector<int64_t> nz_counts(kCoeffOrderMaxSize, 0);
    std::vector<int64_t> gt1_counts(kCoeffOrderMaxSize, 0);

    for (size_t group_index = 0; group_index < frame_dim.num_groups;
         ++group_index) {
      const size_t gx = group_index % frame_dim.xsize_groups;
      const size_t gy = group_index / frame_dim.xsize_groups;
      const Rect rect(gx * kGroupDimInBlocks, gy * kGroupDimInBlocks,
                      kGroupDimInBlocks, kGroupDimInBlocks,
                      frame_dim.xsize_blocks, frame_dim.ysize_blocks);
      const int32_t* rows[3] = {
          ac_image.PlaneRow(0, group_index, 0).ptr32,
          ac_image.PlaneRow(1, group_index, 0).ptr32,
          ac_image.PlaneRow(2, group_index, 0).ptr32,
      };
      size_t ac_offset = 0;
      for (size_t by = 0; by < rect.ysize(); ++by) {
        AcStrategyRow acs_row = ac_strategy.ConstRow(rect, by);
        const int32_t* qf_row =
            raw_quant_field != nullptr ? rect.ConstRow(*raw_quant_field, by)
                                       : nullptr;
        for (size_t bx = 0; bx < rect.xsize(); ++bx) {
          AcStrategy acs = acs_row[bx];
          if (!acs.IsFirstBlock()) continue;
          const size_t llf = acs.covered_blocks_x() * acs.covered_blocks_y();
          const size_t size = kDCTBlockSize * llf;
          const uint8_t ord = kStrategyOrder[acs.RawStrategy()];
          if (ord <= 8 && ((current_used_orders >> ord) & 1u) != 0) {
            size_t cx = acs.covered_blocks_x();
            size_t cy = acs.covered_blocks_y();
            CoefficientLayout(&cy, &cx);
            BlockRef bref;
            bref.qf = (qf_row != nullptr)
                          ? static_cast<uint32_t>(std::max<int32_t>(1, qf_row[bx]))
                          : 1u;
            for (size_t c = 0; c < 3; ++c) {
              const int32_t* blk = rows[c] + ac_offset;
              bref.ptr[c] = blk;
              const size_t order_offset = CoeffOrderOffset(ord, c);
              uint16_t nz = 0;
              for (size_t k = 0; k < size; ++k) {
                const int32_t v = blk[k];
                if (v != 0) {
                  nz_counts[order_offset + k]++;
                  if (v > 1 || v < -1) gt1_counts[order_offset + k]++;
                  const size_t ky = k / (kBlockDim * cx);
                  const size_t kx = k % (kBlockDim * cx);
                  if (ky >= cy || kx >= cx) {
                    ++nz;
                  }
                }
              }
              bref.nz[c] = nz;
            }
            blocks_by_ord[ord].push_back(bref);
          }
          ac_offset += size;
        }
      }
    }

    uint32_t ord0_med_qf = std::numeric_limits<uint32_t>::max();
    if (blocks_by_ord[0].size() >= 256 && raw_quant_field != nullptr) {
      std::vector<uint32_t> qfs;
      qfs.reserve(blocks_by_ord[0].size());
      for (const auto& b : blocks_by_ord[0]) qfs.push_back(b.qf);
      std::nth_element(qfs.begin(), qfs.begin() + qfs.size() / 2, qfs.end());
      ord0_med_qf = qfs[qfs.size() / 2];
    }

    constexpr size_t kSymBins = 40;
    const HybridUintConfig uint_cfg;
    uint8_t sym_lut[513];
    for (int v = -256; v <= 256; ++v) {
      uint32_t tok = 0, nbits = 0, bits = 0;
      uint_cfg.Encode(PackSigned(v), &tok, &nbits, &bits);
      sym_lut[v + 256] = static_cast<uint8_t>(std::min<uint32_t>(tok, kSymBins - 1));
    }

    auto fast_nlog2n = [](uint32_t n) -> double {
      return (n <= 1) ? 0.0 : static_cast<double>(n) * std::log2(static_cast<double>(n));
    };

    constexpr size_t kMaxFineCtxs = 2 * kZeroDensityContextCount;
    constexpr size_t kNumCoarseCtxs = 32;
    std::vector<uint32_t> fine_counts(kMaxFineCtxs * kSymBins, 0);
    std::vector<uint32_t> fine_tot(kMaxFineCtxs, 0);
    std::vector<uint32_t> coarse_counts(kNumCoarseCtxs * kSymBins, 0);
    std::vector<uint32_t> coarse_tot(kNumCoarseCtxs, 0);
    std::vector<uint16_t> dirty_fine;
    dirty_fine.reserve(kMaxFineCtxs);

    auto eval_ac_mdl_bits = [&](uint8_t ord, uint8_t c, size_t llf,
                                size_t log2_llf, size_t sz,
                                const coeff_order_t* cand_ord) -> double {
      const auto& blks = blocks_by_ord[ord];
      if (blks.empty()) return 0.0;
      const bool split_qf = (ord == 0 && ord0_med_qf < std::numeric_limits<uint32_t>::max());
      for (const auto& b : blks) {
        int32_t rem_nz = b.nz[c];
        if (rem_nz == 0) continue;
        const int32_t* JXL_RESTRICT blk = b.ptr[c];
        const size_t qf_off =
            (split_qf && b.qf > ord0_med_qf) ? kZeroDensityContextCount : 0;
        size_t prev = (rem_nz > static_cast<int32_t>(sz / 16)) ? 0 : 1;
        for (size_t k = llf; k < sz && rem_nz != 0; ++k) {
          const int32_t coeff = blk[cand_ord[k]];
          const size_t zd =
              qf_off + ZeroDensityContext(rem_nz, k, llf, log2_llf, prev);
          if (fine_tot[zd] == 0) dirty_fine.push_back(static_cast<uint16_t>(zd));
          fine_tot[zd]++;
          uint32_t sym;
          if (coeff >= -256 && coeff <= 256) {
            sym = sym_lut[coeff + 256];
          } else {
            uint32_t tok = 0, nbits = 0, bits = 0;
            uint_cfg.Encode(PackSigned(coeff), &tok, &nbits, &bits);
            sym = std::min<uint32_t>(tok, kSymBins - 1);
          }
          fine_counts[zd * kSymBins + sym]++;
          prev = (coeff != 0) ? 1 : 0;
          rem_nz -= static_cast<int32_t>(prev);
        }
      }

      std::fill(coarse_counts.begin(), coarse_counts.end(), 0);
      std::fill(coarse_tot.begin(), coarse_tot.end(), 0);
      double total_bits = 0.0;

      auto histo_mdl_bits = [&](const uint32_t* cnts, uint32_t n) -> double {
        if (n == 0) return 0.0;
        double sum_c = 0.0;
        int k_obs = 0;
        for (size_t s = 0; s < kSymBins; ++s) {
          const uint32_t cs = cnts[s];
          if (cs > 0) {
            ++k_obs;
            if (cs > 1) sum_c += fast_nlog2n(cs);
          }
        }
        if (k_obs <= 1) return 0.0;
        const double shannon = fast_nlog2n(n) - sum_c;
        const double mdl =
            0.45 * static_cast<double>(k_obs - 1) * std::log2(1.0 + 0.5 * n);
        return shannon + mdl;
      };

      for (uint16_t zd : dirty_fine) {
        const uint32_t n = fine_tot[zd];
        uint32_t* cnts = &fine_counts[zd * kSymBins];
        const size_t qf_b = (zd >= kZeroDensityContextCount) ? 1 : 0;
        const size_t base_zd = zd - qf_b * kZeroDensityContextCount;
        const size_t cid =
            qf_b * 16 + ((base_zd >> 6) & 7) * 2 + (base_zd & 1);
        coarse_tot[cid] += n;
        for (size_t s = 0; s < kSymBins; ++s) {
          coarse_counts[cid * kSymBins + s] += cnts[s];
        }
        if (n >= 64) {
          total_bits += 0.35 * histo_mdl_bits(cnts, n);
        }
        memset(cnts, 0, kSymBins * sizeof(uint32_t));
        fine_tot[zd] = 0;
      }
      dirty_fine.clear();

      uint32_t fb_counts[2 * kSymBins] = {};
      uint32_t fb_tot[2] = {};
      for (size_t cid = 0; cid < kNumCoarseCtxs; ++cid) {
        if (coarse_tot[cid] >= 8) {
          const double w = (coarse_tot[cid] >= 64) ? 0.65 : 1.00;
          total_bits += w * histo_mdl_bits(&coarse_counts[cid * kSymBins],
                                           coarse_tot[cid]);
        } else if (coarse_tot[cid] > 0) {
          const size_t p = cid & 1;
          fb_tot[p] += coarse_tot[cid];
          for (size_t s = 0; s < kSymBins; ++s) {
            fb_counts[p * kSymBins + s] += coarse_counts[cid * kSymBins + s];
          }
        }
      }
      for (size_t p = 0; p < 2; ++p) {
        if (fb_tot[p] > 0) {
          total_bits += histo_mdl_bits(&fb_counts[p * kSymBins], fb_tot[p]);
        }
      }
      return total_bits;
    };

    std::vector<coeff_order_t> nat_lut_buf;
    std::vector<coeff_order_t> zigzag_tmp;
    auto eval_lehmer_est_bits = [&](AcStrategy acs, size_t sz,
                                    const coeff_order_t* cand_ord,
                                    double* out_bits) -> Status {
      if (nat_lut_buf.size() < sz) nat_lut_buf.resize(sz);
      if (zigzag_tmp.size() < sz) zigzag_tmp.resize(sz);
      acs.ComputeNaturalCoeffOrderLut(nat_lut_buf.data());
      std::vector<Token> ptoks;
      JXL_RETURN_IF_ERROR(EncodeCoeffOrder(cand_ord, acs, &ptoks,
                                           zigzag_tmp.data(), nat_lut_buf));
      if (ptoks.size() <= 1) {
        *out_bits = 1.25;
        return true;
      }
      double b = 2.6;
      for (size_t i = 0; i < ptoks.size(); ++i) {
        const uint32_t v = ptoks[i].value;
        if (v == 0) {
          b += 1.25;
        } else {
          b += 3.1 + 1.12 * static_cast<double>(FloorLog2Nonzero(v));
        }
      }
      *out_bits = b;
      return true;
    };

    struct CandSpec {
      float gran_mul;
      int tail_mul;
      float mag_w;
    };
    static constexpr CandSpec kSpecs[] = {
        {1.00f, 2, 0.00f}, {0.65f, 2, 0.00f}, {1.45f, 2, 0.00f},
        {2.00f, 2, 0.00f}, {1.00f, 1, 0.00f}, {1.00f, 4, 0.00f},
        {0.85f, 2, 0.20f}, {1.25f, 2, 0.20f},
    };

    double ord_ac_gain[9] = {};
    uint32_t cand_used_orders = 0;

    for (int ord = 0; ord <= 8; ++ord) {
      if (((current_used_orders >> ord) & 1u) == 0) continue;
      if (blocks_by_ord[ord].empty()) continue;
      AcStrategy acs = AcStrategy::FromRawStrategy(rep_strategy_for_ord[ord]);
      const size_t llf = acs.covered_blocks_x() * acs.covered_blocks_y();
      const size_t log2_llf = Num0BitsBelowLS1Bit_Nonzero(llf);
      const size_t sz = kDCTBlockSize * llf;
      if (natural_order_buffer.size() < sz) natural_order_buffer.resize(sz);
      acs.ComputeNaturalCoeffOrder(natural_order_buffer.data());

      const size_t num_specs = sizeof(kSpecs) / sizeof(kSpecs[0]);
      const float inv_sqrt_sz = 1.0f / std::sqrt(static_cast<float>(sz));
      const float phase = (sz <= 128) ? 0.0f : 0.5f;

      bool ord_nondefault = false;
      double sum_ac_gain = 0.0;

      for (uint8_t c = 0; c < 3; ++c) {
        const size_t offset = CoeffOrderOffset(ord, c);
        const double nat_ac_bits = eval_ac_mdl_bits(
            ord, c, llf, log2_llf, sz, natural_order_buffer.data());
        double best_tot = nat_ac_bits + 1.25;
        double best_ac = nat_ac_bits;
        std::vector<coeff_order_t> best_ord(
            natural_order_buffer.begin(), natural_order_buffer.begin() + sz);

        std::vector<std::vector<coeff_order_t>> tested_orders;
        tested_orders.push_back(best_ord);

        int64_t max_nz = 0;
        for (size_t i = llf; i < sz; ++i) {
          max_nz = std::max(max_nz, nz_counts[offset + natural_order_buffer[i]]);
        }
        if (max_nz > 0) {
          for (size_t si = 0; si < num_specs; ++si) {
            const CandSpec& sp = kSpecs[si];
            const float step = inv_sqrt_sz * sp.gran_mul;
            int64_t zero_bucket_count = 0;
            size_t last_end = llf;
            const uint64_t top_anchor =
                static_cast<uint64_t>((max_nz * 2) * step + 16.0f);
            for (size_t i = 0; i < sz; ++i) {
              const size_t pos = natural_order_buffer[i];
              pos_and_val[i].pos = static_cast<uint32_t>(pos);
              uint64_t count = 0;
              if (i >= llf) {
                const int64_t nz = nz_counts[offset + pos];
                const float weighted =
                    static_cast<float>(nz) +
                    sp.mag_w * static_cast<float>(gt1_counts[offset + pos]);
                uint64_t nz_bucket =
                    static_cast<uint64_t>(weighted * step + phase);
                if (nz_bucket > 0) {
                  if (zero_bucket_count > 0 &&
                      zero_bucket_count * nz <
                          static_cast<int64_t>(sp.tail_mul) *
                              static_cast<int64_t>(i + 1 - last_end)) {
                    nz_bucket = 0;
                  } else {
                    last_end = i + 1;
                  }
                }
                if (nz_bucket == 0) {
                  ++zero_bucket_count;
                }
                count = top_anchor + 2 - std::min<uint64_t>(top_anchor, nz_bucket);
              }
              pos_and_val[i].count_and_idx = (count << 16) | i;
            }
            std::sort(pos_and_val, pos_and_val + sz,
                      [](const PosAndCount& a, const PosAndCount& b) {
                        return a.count_and_idx < b.count_and_idx;
                      });

            std::vector<coeff_order_t> cand_ord(sz);
            for (size_t i = 0; i < sz; ++i) cand_ord[i] = pos_and_val[i].pos;

            bool dup = false;
            for (const auto& prev_o : tested_orders) {
              if (memcmp(prev_o.data(), cand_ord.data(),
                         sz * sizeof(coeff_order_t)) == 0) {
                dup = true;
                break;
              }
            }
            if (dup) continue;
            tested_orders.push_back(cand_ord);

            double lehmer_bits = 0.0;
            JXL_RETURN_IF_ERROR(
                eval_lehmer_est_bits(acs, sz, cand_ord.data(), &lehmer_bits));
            const double ac_bits =
                eval_ac_mdl_bits(ord, c, llf, log2_llf, sz, cand_ord.data());
            if (ac_bits + lehmer_bits + 0.5 < best_tot) {
              best_tot = ac_bits + lehmer_bits;
              best_ac = ac_bits;
              best_ord = std::move(cand_ord);
            }
          }
        }

        memcpy(&order[offset], best_ord.data(), sz * sizeof(*order));
        if (memcmp(best_ord.data(), natural_order_buffer.data(),
                   sz * sizeof(*order)) != 0) {
          ord_nondefault = true;
          sum_ac_gain += (nat_ac_bits - best_ac);
        }
      }

      if (ord_nondefault && sum_ac_gain > 1.0) {
        cand_used_orders |= (1u << ord);
        ord_ac_gain[ord] = sum_ac_gain;
      } else {
        for (uint8_t c = 0; c < 3; ++c) {
          memcpy(&order[CoeffOrderOffset(ord, c)], natural_order_buffer.data(),
                 sz * sizeof(*order));
        }
      }
    }

    auto best_hdr_for_mask = [&](uint32_t base_mask,
                                 uint32_t* out_written_mask,
                                 size_t* out_bits) -> Status {
      if (base_mask == 0) {
        *out_written_mask = 0;
        *out_bits = 2;
        return true;
      }
      uint32_t masks[3] = {base_mask, 0, 0};
      size_t n_masks = 1;
      if ((base_mask & ~0x13u) == 0 && base_mask != 0x13u) {
        masks[n_masks++] = 0x13u;
      }
      if ((base_mask & ~0x5Fu) == 0 && base_mask != 0x5Fu) {
        masks[n_masks++] = 0x5Fu;
      }
      size_t best_b = std::numeric_limits<size_t>::max();
      uint32_t best_m = base_mask;
      for (size_t mi = 0; mi < n_masks; ++mi) {
        size_t b = 0;
        JXL_RETURN_IF_ERROR(EstimateOrderPayloadBits(
            memory_manager, static_cast<uint16_t>(masks[mi]), order, &b));
        if (b < best_b) {
          best_b = b;
          best_m = masks[mi];
        }
      }
      *out_written_mask = best_m;
      *out_bits = best_b;
      return true;
    };

    uint32_t cur_written_mask = 0;
    size_t cur_hdr_bits = 2;
    JXL_RETURN_IF_ERROR(
        best_hdr_for_mask(cand_used_orders, &cur_written_mask, &cur_hdr_bits));

    while (cand_used_orders != 0) {
      double worst_net = std::numeric_limits<double>::max();
      int worst_ord = -1;
      uint32_t worst_sub_written = 0;
      size_t worst_sub_bits = 2;

      for (int ord = 0; ord <= 8; ++ord) {
        if (((cand_used_orders >> ord) & 1u) == 0) continue;
        AcStrategy acs = AcStrategy::FromRawStrategy(rep_strategy_for_ord[ord]);
        const size_t sz =
            kDCTBlockSize * acs.covered_blocks_x() * acs.covered_blocks_y();
        if (natural_order_buffer.size() < sz) natural_order_buffer.resize(sz);
        acs.ComputeNaturalCoeffOrder(natural_order_buffer.data());

        std::vector<coeff_order_t> saved_ord(3 * sz);
        for (size_t c = 0; c < 3; ++c) {
          const size_t off = CoeffOrderOffset(ord, c);
          memcpy(&saved_ord[c * sz], &order[off], sz * sizeof(*order));
          memcpy(&order[off], natural_order_buffer.data(), sz * sizeof(*order));
        }

        const uint32_t sub_mask = cand_used_orders & ~(1u << ord);
        uint32_t sub_written = 0;
        size_t sub_bits = 2;
        Status st = best_hdr_for_mask(sub_mask, &sub_written, &sub_bits);
        for (size_t c = 0; c < 3; ++c) {
          memcpy(&order[CoeffOrderOffset(ord, c)], &saved_ord[c * sz],
                 sz * sizeof(*order));
        }
        JXL_RETURN_IF_ERROR(st);

        const double marginal_hdr =
            static_cast<double>(cur_hdr_bits) - static_cast<double>(sub_bits);
        const double net = ord_ac_gain[ord] - marginal_hdr;
        if (net < worst_net) {
          worst_net = net;
          worst_ord = ord;
          worst_sub_written = sub_written;
          worst_sub_bits = sub_bits;
        }
      }

      if (worst_ord >= 0 && worst_net <= 1.0) {
        AcStrategy acs =
            AcStrategy::FromRawStrategy(rep_strategy_for_ord[worst_ord]);
        const size_t sz =
            kDCTBlockSize * acs.covered_blocks_x() * acs.covered_blocks_y();
        if (natural_order_buffer.size() < sz) natural_order_buffer.resize(sz);
        acs.ComputeNaturalCoeffOrder(natural_order_buffer.data());
        for (size_t c = 0; c < 3; ++c) {
          memcpy(&order[CoeffOrderOffset(worst_ord, c)],
                 natural_order_buffer.data(), sz * sizeof(*order));
        }
        cand_used_orders &= ~(1u << worst_ord);
        ord_ac_gain[worst_ord] = 0.0;
        cur_written_mask = worst_sub_written;
        cur_hdr_bits = worst_sub_bits;
      } else {
        break;
      }
    }

    double total_ac_gain = 0.0;
    for (int ord = 0; ord <= 8; ++ord) {
      if ((cand_used_orders >> ord) & 1u) total_ac_gain += ord_ac_gain[ord];
    }
    if (cand_used_orders != 0 &&
        total_ac_gain > static_cast<double>(cur_hdr_bits) - 2.0 + 1.0) {
      all_used_orders |= cur_written_mask;
    } else {
      for (int ord = 0; ord <= 8; ++ord) {
        if (((cand_used_orders >> ord) & 1u) == 0) continue;
        AcStrategy acs = AcStrategy::FromRawStrategy(rep_strategy_for_ord[ord]);
        const size_t sz =
            kDCTBlockSize * acs.covered_blocks_x() * acs.covered_blocks_y();
        if (natural_order_buffer.size() < sz) natural_order_buffer.resize(sz);
        acs.ComputeNaturalCoeffOrder(natural_order_buffer.data());
        for (size_t c = 0; c < 3; ++c) {
          memcpy(&order[CoeffOrderOffset(ord, c)], natural_order_buffer.data(),
                 sz * sizeof(*order));
        }
      }
    }
    return true;
  }

  std::vector<int64_t> num_zeros(kCoeffOrderMaxSize);
  // If compressing at high speed (> kSquirrel) and only using 8x8 DCTs, only
  // consider a subset of blocks.
  double block_fraction = 1.0f;
  if ((speed > SpeedTier::kSquirrel || raw_quant_field == nullptr) &&
      speed >= SpeedTier::kSquirrel && current_used_orders == 1) {
    block_fraction = 0.5f;
  }
  // No need to compute number of zero coefficients if all orders are the
  // default.
  if (current_used_orders != 0) {
    uint64_t threshold =
        (std::numeric_limits<uint64_t>::max() >> 32) * block_fraction;
    uint64_t s[2] = {static_cast<uint64_t>(0x94D049BB133111EBull),
                     static_cast<uint64_t>(0xBF58476D1CE4E5B9ull)};
    // Xorshift128+ adapted from xorshift128+-inl.h
    auto use_sample = [&]() {
      auto s1 = s[0];
      const auto s0 = s[1];
      const auto bits = s1 + s0;  // b, c
      s[0] = s0;
      s1 ^= s1 << 23;
      s1 ^= s0 ^ (s1 >> 18) ^ (s0 >> 5);
      s[1] = s1;
      return (bits >> 32) <= threshold;
    };

    // Count number of zero coefficients, separately for each DCT band.
    // TODO(veluca): precompute when doing DCT.
    for (size_t group_index = 0; group_index < frame_dim.num_groups;
         group_index++) {
      const size_t gx = group_index % frame_dim.xsize_groups;
      const size_t gy = group_index / frame_dim.xsize_groups;
      const Rect rect(gx * kGroupDimInBlocks, gy * kGroupDimInBlocks,
                      kGroupDimInBlocks, kGroupDimInBlocks,
                      frame_dim.xsize_blocks, frame_dim.ysize_blocks);
      ConstACPtr rows[3];
      ACType type = ac_image.Type();
      for (size_t c = 0; c < 3; c++) {
        rows[c] = ac_image.PlaneRow(c, group_index, 0);
      }
      size_t ac_offset = 0;

      // TODO(veluca): SIMDfy.
      for (size_t by = 0; by < rect.ysize(); ++by) {
        AcStrategyRow acs_row = ac_strategy.ConstRow(rect, by);
        for (size_t bx = 0; bx < rect.xsize(); ++bx) {
          AcStrategy acs = acs_row[bx];
          if (!acs.IsFirstBlock()) continue;
          size_t size = kDCTBlockSize << acs.log2_covered_blocks();
          if (use_sample()) {
            for (size_t c = 0; c < 3; ++c) {
              const size_t order_offset =
                  CoeffOrderOffset(kStrategyOrder[acs.RawStrategy()], c);
              if (type == ACType::k16) {
                for (size_t k = 0; k < size; k++) {
                  bool is_zero = rows[c].ptr16[ac_offset + k] == 0;
                  num_zeros[order_offset + k] += is_zero ? 1 : 0;
                }
              } else {
                for (size_t k = 0; k < size; k++) {
                  bool is_zero = rows[c].ptr32[ac_offset + k] == 0;
                  num_zeros[order_offset + k] += is_zero ? 1 : 0;
                }
              }
            }
          }
          ac_offset += size;
        }
      }
    }
  }
  struct PosAndCount {
    uint32_t pos;
    // Saving index breaks the ties for non-stable sort
    uint64_t count_and_idx;
  };
  size_t mem_bytes = AcStrategy::kMaxCoeffArea * sizeof(PosAndCount);
  JXL_ASSIGN_OR_RETURN(auto mem,
                       AlignedMemory::Create(memory_manager, mem_bytes));

  std::vector<coeff_order_t> natural_order_buffer;

  const bool enable_order_rdo =
      frame_dim.xsize_blocks * frame_dim.ysize_blocks > 64;
  uint16_t computed = 0;
  for (uint8_t o = 0; o < AcStrategy::kNumValidStrategies; ++o) {
    uint8_t ord = kStrategyOrder[o];
    if (computed & (1 << ord)) continue;
    computed |= 1 << ord;
    AcStrategy acs = AcStrategy::FromRawStrategy(o);
    size_t sz = kDCTBlockSize * acs.covered_blocks_x() * acs.covered_blocks_y();
    // Expected maximal size is 256 x 256.
    JXL_DASSERT(sz <= (1 << 16));

    // Do nothing for transforms that don't appear.
    if ((1 << ord) & ~current_used_acs) continue;

    // Do nothing if we already committed to this custom order previously.
    if ((1 << ord) & prev_used_acs) continue;
    if ((1 << ord) & all_used_orders) continue;

    if (natural_order_buffer.size() < sz) natural_order_buffer.resize(sz);
    acs.ComputeNaturalCoeffOrder(natural_order_buffer.data());

    // Ensure natural coefficient order is not permuted if the order is
    // not transmitted.
    if ((1 << ord) & ~current_used_orders) {
      for (size_t c = 0; c < 3; c++) {
        size_t offset = CoeffOrderOffset(ord, c);
        JXL_ENSURE(CoeffOrderOffset(ord, c + 1) - offset == sz);
        memcpy(&order[offset], natural_order_buffer.data(),
               sz * sizeof(*order));
      }
      continue;
    }

    size_t cx = acs.covered_blocks_x();
    size_t cy = acs.covered_blocks_y();
    CoefficientLayout(&cy, &cx);
    bool is_nondefault = false;
    for (uint8_t c = 0; c < 3; c++) {
      // Apply zig-zag order.
      PosAndCount* pos_and_val = mem.address<PosAndCount>();
      size_t offset = CoeffOrderOffset(ord, c);
      JXL_ENSURE(CoeffOrderOffset(ord, c + 1) - offset == sz);
      // Ensure LLFs are first in the order.
      for (size_t iy = 0; iy < cy; iy++) {
        for (size_t ix = 0; ix < cx; ix++) {
          num_zeros[offset + iy * kBlockDim * cx + ix] = -1;
        }
      }
      size_t llf = acs.covered_blocks_x() * acs.covered_blocks_y();
      int64_t max_zeros = 0;
      for (size_t i = llf; i < sz; ++i) {
        max_zeros = std::max(max_zeros, num_zeros[offset + natural_order_buffer[i]]);
      }
      float inv_sqrt_sz = 1.0f / std::sqrt(sz);
      const float phase = (sz <= 128) ? 0.0f : 0.5f;
      int64_t zero_bucket_count = 0;
      size_t last_end = llf;
      for (size_t i = 0; i < sz; ++i) {
        size_t pos = natural_order_buffer[i];
        pos_and_val[i].pos = pos;
        uint64_t count = 0;
        if (!enable_order_rdo) {
          count = num_zeros[offset + pos] * inv_sqrt_sz + 0.1f;
        } else if (i >= llf) {
          // Anchor quantization at max_zeros (0 non-zeros) so that isolated
          // high-frequency non-zeros stay in the unpermuted tail.
          int64_t nonzeros = max_zeros - num_zeros[offset + pos];
          uint64_t nz_bucket =
              static_cast<uint64_t>(nonzeros * inv_sqrt_sz + phase);
          if (nz_bucket > 0) {
            if (zero_bucket_count > 0 &&
                zero_bucket_count * nonzeros <
                    2 * static_cast<int64_t>(i + 1 - last_end)) {
              nz_bucket = 0;
            } else {
              last_end = i + 1;
            }
          }
          if (nz_bucket == 0) {
            ++zero_bucket_count;
          }
          count = static_cast<uint64_t>(max_zeros + 2) - nz_bucket;
        }
        JXL_DASSERT(count < (uint64_t{1} << 48));
        pos_and_val[i].count_and_idx = (count << 16) | i;
      }

      // Stable-sort -> elements with same number of zeros will preserve their
      // order.
      auto comparator = [](const PosAndCount& a, const PosAndCount& b) -> bool {
        return a.count_and_idx < b.count_and_idx;
      };
      std::sort(pos_and_val, pos_and_val + sz, comparator);

      // Per-channel RDO gate: require total scan-index advancement of non-zero
      // coefficients to outweigh the Lehmer permutation span (end - llf).
      int64_t scan_gain = 0;
      size_t end = llf;
      for (size_t i = llf; i < sz; ++i) {
        const size_t nat_pos = natural_order_buffer[i];
        const size_t ord_pos = pos_and_val[i].pos;
        if (nat_pos != ord_pos) {
          end = i + 1;
          const int64_t nz_nat = max_zeros - num_zeros[offset + nat_pos];
          const int64_t nz_ord = max_zeros - num_zeros[offset + ord_pos];
          scan_gain += static_cast<int64_t>(i) * (nz_nat - nz_ord);
        }
      }
      if (!enable_order_rdo ||
          (end > llf &&
           scan_gain >= static_cast<int64_t>(2 * (end - llf) * block_fraction))) {
        for (size_t i = 0; i < sz; ++i) {
          order[offset + i] = pos_and_val[i].pos;
          is_nondefault |= natural_order_buffer[i] != pos_and_val[i].pos;
        }
      } else {
        memcpy(&order[offset], natural_order_buffer.data(),
               sz * sizeof(*order));
      }
    }
    if (!is_nondefault) {
      current_used_orders &= ~(1 << ord);
    }
  }
  all_used_orders |= current_used_orders;
  return true;
}

Status EncodePermutation(const coeff_order_t* JXL_RESTRICT order, size_t skip,
                         size_t size, BitWriter* writer, LayerType layer,
                         AuxOut* aux_out) {
  JxlMemoryManager* memory_manager = writer->memory_manager();
  std::vector<std::vector<Token>> tokens(1);
  JXL_RETURN_IF_ERROR(TokenizePermutation(order, skip, size, tokens.data()));
  EntropyEncodingData codes;
  JXL_ASSIGN_OR_RETURN(
      size_t cost, BuildAndEncodeHistograms(memory_manager, HistogramParams(),
                                            kPermutationContexts, tokens,
                                            &codes, writer, layer, aux_out));
  (void)cost;
  JXL_RETURN_IF_ERROR(WriteTokens(tokens[0], codes, 0, writer, layer, aux_out));
  return true;
}

Status EncodeCoeffOrders(uint16_t used_orders,
                         const coeff_order_t* JXL_RESTRICT order,
                         BitWriter* writer, LayerType layer,
                         AuxOut* JXL_RESTRICT aux_out) {
  JxlMemoryManager* memory_manager = writer->memory_manager();
  size_t mem_bytes = AcStrategy::kMaxCoeffArea * sizeof(coeff_order_t);
  JXL_ASSIGN_OR_RETURN(auto mem,
                       AlignedMemory::Create(memory_manager, mem_bytes));
  uint16_t computed = 0;
  std::vector<std::vector<Token>> tokens(1);
  std::vector<coeff_order_t> natural_order_lut;
  for (uint8_t o = 0; o < AcStrategy::kNumValidStrategies; ++o) {
    uint8_t ord = kStrategyOrder[o];
    if (computed & (1 << ord)) continue;
    computed |= 1 << ord;
    if ((used_orders & (1 << ord)) == 0) continue;
    AcStrategy acs = AcStrategy::FromRawStrategy(o);
    const size_t llf = acs.covered_blocks_x() * acs.covered_blocks_y();
    const size_t size = kDCTBlockSize * llf;
    if (natural_order_lut.size() < size) natural_order_lut.resize(size);
    acs.ComputeNaturalCoeffOrderLut(natural_order_lut.data());
    for (size_t c = 0; c < 3; c++) {
      JXL_RETURN_IF_ERROR(
          EncodeCoeffOrder(&order[CoeffOrderOffset(ord, c)], acs, tokens.data(),
                           mem.address<coeff_order_t>(), natural_order_lut));
    }
  }
  // Do not write anything if no order is used.
  if (used_orders != 0) {
    HistogramParams best_hp;
    size_t best_bits = std::numeric_limits<size_t>::max();
    HistogramParams hp_cands[7];
    hp_cands[1].lz77_method = HistogramParams::LZ77Method::kRLE;
    hp_cands[1].rdo_context_map = true;
    hp_cands[2].lz77_method = HistogramParams::LZ77Method::kNone;
    hp_cands[2].rdo_context_map = true;
    hp_cands[3].lz77_method = HistogramParams::LZ77Method::kRLE;
    hp_cands[3].rdo_context_map = true;
    hp_cands[3].force_huffman = true;
    hp_cands[4].lz77_method = HistogramParams::LZ77Method::kNone;
    hp_cands[4].rdo_context_map = true;
    hp_cands[4].force_huffman = true;
    hp_cands[5].lz77_method = HistogramParams::LZ77Method::kOptc256;
    hp_cands[5].rdo_context_map = true;
    hp_cands[6].lz77_method = HistogramParams::LZ77Method::kOptc256;
    hp_cands[6].rdo_context_map = true;
    hp_cands[6].force_huffman = true;
    const size_t num_cands = (tokens[0].size() >= 16) ? 7 : 5;
    for (size_t ci = 0; ci < num_cands; ++ci) {
      BitWriter cand_bw{memory_manager};
      std::vector<std::vector<Token>> cand_toks = tokens;
      EntropyEncodingData cand_codes;
      JXL_ASSIGN_OR_RETURN(
          size_t ccost,
          BuildAndEncodeHistograms(memory_manager, hp_cands[ci],
                                   kPermutationContexts, cand_toks, &cand_codes,
                                   &cand_bw, LayerType::Header, nullptr));
      (void)ccost;
      JXL_RETURN_IF_ERROR(WriteTokens(cand_toks[0], cand_codes, 0, &cand_bw,
                                      LayerType::Header, nullptr));
      if (cand_bw.BitsWritten() < best_bits) {
        best_bits = cand_bw.BitsWritten();
        best_hp = hp_cands[ci];
      }
    }
    EntropyEncodingData codes;
    JXL_ASSIGN_OR_RETURN(
        size_t cost, BuildAndEncodeHistograms(memory_manager, best_hp,
                                              kPermutationContexts, tokens,
                                              &codes, writer, layer, aux_out));
    (void)cost;
    JXL_RETURN_IF_ERROR(
        WriteTokens(tokens[0], codes, 0, writer, layer, aux_out));
  }
  return true;
}

}  // namespace jxl
