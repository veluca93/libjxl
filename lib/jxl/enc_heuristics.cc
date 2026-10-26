// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_heuristics.h"

#include <jxl/cms_interface.h>
#include <jxl/memory_manager.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "lib/jxl/ac_context.h"
#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/override.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/butteraugli/butteraugli.h"
#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/coeff_order.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/common.h"
#include "lib/jxl/dct_util.h"
#include "lib/jxl/dec_cache.h"
#include "lib/jxl/dec_group.h"
#include "lib/jxl/dec_noise.h"
#include "lib/jxl/dec_xyb.h"
#include "lib/jxl/enc_ac_strategy.h"
#include "lib/jxl/enc_adaptive_quantization.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_chroma_from_luma.h"
#include "lib/jxl/enc_coeff_order.h"
#include "lib/jxl/enc_entropy_coder.h"
#include "lib/jxl/enc_gaborish.h"
#include "lib/jxl/enc_group.h"
#include "lib/jxl/enc_modular.h"
#include "lib/jxl/enc_noise.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/enc_patch_dictionary.h"
#include "lib/jxl/enc_quant_weights.h"
#include "lib/jxl/enc_splines.h"
#include "lib/jxl/entropy_coder.h"
#include "lib/jxl/epf.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/pack_signed.h"
#include "lib/jxl/passes_state.h"
#include "lib/jxl/quant_weights.h"
#include "lib/jxl/render_pipeline/render_pipeline.h"

namespace jxl {

struct AuxOut;

void FindBestBlockEntropyModel(const CompressParams& cparams, const ImageI& rqf,
                               const AcStrategyImage& ac_strategy,
                               BlockCtxMap* block_ctx_map) {
  if (cparams.decoding_speed_tier >= 1) {
    static constexpr uint8_t kSimpleCtxMap[] = {
        // Cluster all blocks together
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  //
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  //
    };
    static_assert(
        3 * kNumOrders == sizeof(kSimpleCtxMap) / sizeof *kSimpleCtxMap,
        "Update simple context map");

    auto& bcm = *block_ctx_map;
    bcm.ctx_map.assign(std::begin(kSimpleCtxMap), std::end(kSimpleCtxMap));
    bcm.num_ctxs = 2;
    bcm.num_dc_ctxs = 1;
    return;
  }
  if (cparams.speed_tier >= SpeedTier::kFalcon) {
    return;
  }
  // No need to change context modeling for small images.
  size_t tot = rqf.xsize() * rqf.ysize();
  size_t size_for_ctx_model = (1 << 10) * cparams.butteraugli_distance;
  if (tot < size_for_ctx_model) return;

  struct OccCounters {
    // count the occurrences of each qf value and each strategy type.
    OccCounters(const ImageI& rqf, const AcStrategyImage& ac_strategy) {
      for (size_t y = 0; y < rqf.ysize(); y++) {
        const int32_t* qf_row = rqf.Row(y);
        AcStrategyRow acs_row = ac_strategy.ConstRow(y);
        for (size_t x = 0; x < rqf.xsize(); x++) {
          int ord = kStrategyOrder[acs_row[x].RawStrategy()];
          int qf = qf_row[x] - 1;
          qf_counts[qf]++;
          qf_ord_counts[ord][qf]++;
          ord_counts[ord]++;
        }
      }
    }

    size_t qf_counts[256] = {};
    size_t qf_ord_counts[kNumOrders][256] = {};
    size_t ord_counts[kNumOrders] = {};
  };
  // The OccCounters struct is too big to allocate on the stack.
  std::unique_ptr<OccCounters> counters(new OccCounters(rqf, ac_strategy));

  // Splitting the context model according to the quantization field seems to
  // mostly benefit only large images.
  size_t size_for_qf_split = (1 << 13) * cparams.butteraugli_distance;
  size_t num_qf_segments = tot < size_for_qf_split ? 1 : 2;
  std::vector<uint32_t>& qft = block_ctx_map->qf_thresholds;
  qft.clear();
  // Divide the quant field in up to num_qf_segments segments.
  size_t cumsum = 0;
  size_t next = 1;
  size_t last_cut = 256;
  size_t cut = tot * next / num_qf_segments;
  for (uint32_t j = 0; j < 256; j++) {
    const size_t prev_cumsum = cumsum;
    cumsum += counters->qf_counts[j];
    if (cumsum > cut) {
      if (prev_cumsum > 0 && prev_cumsum < tot) {
        qft.push_back(j);
      }
      last_cut = j;
      while (cumsum > cut) {
        next++;
        cut = tot * next / num_qf_segments;
      }
    } else if (next > qft.size() + 1) {
      if (j - 1 == last_cut && prev_cumsum > 0 && prev_cumsum < tot) {
        qft.push_back(j);
      }
    }
  }

  // Count the occurrences of each segment.
  std::vector<size_t> counts(kNumOrders * (qft.size() + 1));
  size_t qft_pos = 0;
  for (size_t j = 0; j < 256; j++) {
    if (qft_pos < qft.size() && j == qft[qft_pos]) {
      qft_pos++;
    }
    for (size_t i = 0; i < kNumOrders; i++) {
      counts[qft_pos + i * (qft.size() + 1)] += counters->qf_ord_counts[i][j];
    }
  }

  // Repeatedly merge the lowest-count pair.
  std::vector<uint8_t> remap((qft.size() + 1) * kNumOrders);
  std::iota(remap.begin(), remap.end(), 0);
  std::vector<uint8_t> clusters(remap);
  size_t nb_clusters =
      Clamp1(static_cast<int>(tot / size_for_ctx_model / 2), 2, 9);
  size_t nb_clusters_chroma =
      Clamp1(static_cast<int>(tot / size_for_ctx_model / 3), 1, 5);
  // This is O(n^2 log n), but n is small.
  while (clusters.size() > nb_clusters ||
         (clusters.size() > 1 &&
          counts[*std::min_element(
              clusters.begin(), clusters.end(),
              [&](int a, int b) { return counts[a] < counts[b]; })] == 0)) {
    std::sort(clusters.begin(), clusters.end(),
              [&](int a, int b) { return counts[a] > counts[b]; });
    counts[clusters[clusters.size() - 2]] += counts[clusters.back()];
    counts[clusters.back()] = 0;
    remap[clusters.back()] = clusters[clusters.size() - 2];
    clusters.pop_back();
  }
  for (size_t i = 0; i < remap.size(); i++) {
    while (remap[remap[i]] != remap[i]) {
      remap[i] = remap[remap[i]];
    }
  }
  // Sort remaining cluster roots by descending count so the largest clusters
  // receive the lowest indices (and thus remain unmerged for chroma below).
  std::sort(clusters.begin(), clusters.end(),
            [&](int a, int b) { return counts[a] > counts[b]; });
  // Relabel starting from 0 in descending cluster size order.
  std::vector<uint8_t> remap_remap(remap.size(), remap.size());
  size_t num = 0;
  for (uint8_t c : clusters) {
    remap_remap[c] = num++;
  }
  for (size_t i = 0; i < remap.size(); i++) {
    if (remap_remap[remap[i]] == remap.size()) {
      remap_remap[remap[i]] = num++;
    }
    remap[i] = remap_remap[remap[i]];
  }
  // Write the block context map.
  auto& ctx_map = block_ctx_map->ctx_map;
  ctx_map = remap;
  ctx_map.resize(remap.size() * 3);
  // for chroma, only use up to nb_clusters_chroma separate block contexts
  // (those for the biggest clusters)
  for (size_t i = remap.size(); i < remap.size() * 3; i++) {
    ctx_map[i] = num + Clamp1(static_cast<int>(remap[i % remap.size()]), 0,
                              static_cast<int>(nb_clusters_chroma) - 1);
  }
  block_ctx_map->num_ctxs =
      *std::max_element(ctx_map.begin(), ctx_map.end()) + 1;
}

Status FindBestBlockEntropyModelFromCoeffs(const FrameHeader& frame_header,
                                           const coeff_order_t* coeff_orders,
                                           PassesEncoderState* enc_state,
                                           BlockCtxMap* block_ctx_map) {
  const CompressParams& cparams = enc_state->cparams;
  if (cparams.decoding_speed_tier >= 1) {
    static constexpr uint8_t kSimpleCtxMap[] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  //
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  //
    };
    block_ctx_map->qf_thresholds.clear();
    for (auto& dct : block_ctx_map->dc_thresholds) dct.clear();
    block_ctx_map->ctx_map.assign(std::begin(kSimpleCtxMap),
                                  std::end(kSimpleCtxMap));
    block_ctx_map->num_ctxs = 2;
    block_ctx_map->num_dc_ctxs = 1;
    return true;
  }
  if (cparams.speed_tier >= SpeedTier::kFalcon || enc_state->coeffs.empty()) {
    return true;
  }

  PassesSharedState& shared = enc_state->shared;
  const FrameDimensions& frame_dim = shared.frame_dim;
  const ImageI& rqf = shared.raw_quant_field;
  const AcStrategyImage& ac_strategy = shared.ac_strategy;
  const ImageB& qdc = shared.quant_dc;
  const YCbCrChromaSubsampling& cs = frame_header.chroma_subsampling;
  const size_t tot_blocks = rqf.xsize() * rqf.ysize();
  if (tot_blocks < 64) {
    return true;
  }

  const ACImage& coeffs = *enc_state->coeffs[0];
  JXL_ENSURE(coeffs.Type() == ACType::k32);
  JxlMemoryManager* memory_manager = enc_state->memory_manager();

  std::vector<double> nlog2n_tab(8192, 0.0);
  for (uint32_t i = 2; i < 8192; ++i) {
    nlog2n_tab[i] = static_cast<double>(i) * std::log2(static_cast<double>(i));
  }
  const auto fast_nlog2n = [&](uint32_t n) -> double {
    if (n < 8192) return nlog2n_tab[n];
    return static_cast<double>(n) * std::log2(static_cast<double>(n));
  };

  // Step 1: Collect per-(c_slice, ord, qf) 16-bin token summaries on first
  // blocks and pick optimal qf_thresholds via 1D dynamic programming.
  const double d_target = std::max<double>(0.25, cparams.butteraugli_distance);
  const double size_for_ctx_model = 1024.0 * d_target;
  const size_t max_luma_ctxs =
      (tot_blocks < size_for_ctx_model)
          ? 2
          : static_cast<size_t>(
                Clamp1(static_cast<int>(tot_blocks / size_for_ctx_model / 2.0),
                       2, 10));
  const size_t max_chroma_ctxs =
      (tot_blocks < size_for_ctx_model)
          ? 1
          : static_cast<size_t>(
                Clamp1(static_cast<int>(tot_blocks / size_for_ctx_model / 3.0),
                       1, 6));

  const size_t num_dc_ctxs = std::max<size_t>(1, block_ctx_map->num_dc_ctxs);
  const double size_for_qf_split = 8192.0 * d_target;
  const size_t max_q_segs_by_size =
      (tot_blocks < size_for_qf_split)
          ? 1
          : (tot_blocks < 4.0 * size_for_qf_split ? 2 : 3);
  const size_t max_q_segs =
      std::min<size_t>(max_q_segs_by_size, 64 / num_dc_ctxs);
  constexpr size_t kQfBins = 16;
  std::vector<uint32_t> qf_hist(3 * kNumOrders * 257 * kQfBins, 0);
  std::array<uint32_t, 257> qf_block_counts{};

  for (size_t group_index = 0; group_index < frame_dim.num_groups;
       ++group_index) {
    const Rect grect = frame_dim.BlockGroupRect(group_index);
    const int32_t* JXL_RESTRICT ac_rows[3] = {
        coeffs.PlaneRow(0, group_index, 0).ptr32,
        coeffs.PlaneRow(1, group_index, 0).ptr32,
        coeffs.PlaneRow(2, group_index, 0).ptr32,
    };
    size_t offset[3] = {};
    for (size_t by = 0; by < grect.ysize(); ++by) {
      const size_t sby[3] = {by >> cs.VShift(0), by >> cs.VShift(1),
                             by >> cs.VShift(2)};
      const int32_t* JXL_RESTRICT row_qf = grect.ConstRow(rqf, by);
      AcStrategyRow acs_row = ac_strategy.ConstRow(grect, by);
      for (size_t bx = 0; bx < grect.xsize(); ++bx) {
        AcStrategy acs = acs_row[bx];
        if (!acs.IsFirstBlock()) continue;
        const size_t sbx[3] = {bx >> cs.HShift(0), bx >> cs.HShift(1),
                               bx >> cs.HShift(2)};
        const size_t covered_blocks =
            acs.covered_blocks_x() * acs.covered_blocks_y();
        const size_t size = covered_blocks * kDCTBlockSize;
        const int ord = kStrategyOrder[acs.RawStrategy()];
        for (int c : {1, 0, 2}) {
          if ((sbx[c] << cs.HShift(c)) != bx ||
              (sby[c] << cs.VShift(c)) != by) {
            continue;
          }
          const int32_t* JXL_RESTRICT block = ac_rows[c] + offset[c];
          const coeff_order_t* JXL_RESTRICT order =
              &coeff_orders[CoeffOrderOffset(ord, c)];
          const int32_t qf_val = Clamp1<int32_t>(row_qf[sbx[c]], 1, 256);
          if (c == 1) qf_block_counts[qf_val]++;
          const size_t c_slice = (c < 2) ? (c ^ 1) : 2;
          uint32_t* JXL_RESTRICT h =
              &qf_hist[((c_slice * kNumOrders + ord) * 257 + qf_val) * kQfBins];

          int32_t nzeros = 0;
          for (size_t k = covered_blocks; k < size; ++k) {
            nzeros += (block[order[k]] != 0);
          }
          const uint32_t nz_bin =
              (nzeros == 0)
                  ? 0
                  : std::min<uint32_t>(
                        7, 1 + FloorLog2Nonzero(static_cast<uint32_t>(nzeros)));
          h[nz_bin]++;

          int32_t rem_nz = nzeros;
          for (size_t k = covered_blocks; k < size && rem_nz > 0; ++k) {
            const int32_t coeff = block[order[k]];
            if (coeff == 0) {
              h[8]++;
            } else {
              rem_nz--;
              const uint32_t abs_c = static_cast<uint32_t>(std::abs(coeff));
              const uint32_t ac_bin =
                  (abs_c == 1)
                      ? 9
                      : std::min<uint32_t>(15, 9 + FloorLog2Nonzero(abs_c));
              h[ac_bin]++;
            }
          }
          offset[c] += size;
        }
      }
    }
  }

  std::vector<uint32_t> active_qfs;
  active_qfs.reserve(32);
  for (uint32_t v = 1; v <= 256; ++v) {
    if (qf_block_counts[v] > 0) active_qfs.push_back(v);
  }

  block_ctx_map->qf_thresholds.clear();
  if (active_qfs.size() >= 2 && max_q_segs >= 2) {
    const size_t K = active_qfs.size();
    const size_t num_co = 3 * kNumOrders;
    std::vector<uint32_t> pref(num_co * (K + 1) * kQfBins, 0);
    std::vector<uint32_t> pref_blocks(K + 1, 0);
    for (size_t idx = 0; idx < K; ++idx) {
      const uint32_t v = active_qfs[idx];
      pref_blocks[idx + 1] = pref_blocks[idx] + qf_block_counts[v];
      for (size_t co = 0; co < num_co; ++co) {
        const uint32_t* src = &qf_hist[(co * 257 + v) * kQfBins];
        const uint32_t* prv = &pref[(co * (K + 1) + idx) * kQfBins];
        uint32_t* dst = &pref[(co * (K + 1) + idx + 1) * kQfBins];
        for (size_t b = 0; b < kQfBins; ++b) {
          dst[b] = prv[b] + src[b];
        }
      }
    }

    constexpr uint32_t kMinSegBlocks = 12;
    constexpr double kQfNullBias = 0.70;
    constexpr double kInf = 1e30;
    std::vector<double> seg_cost(K * K, kInf);
    for (size_t i = 0; i < K; ++i) {
      for (size_t j = i; j < K; ++j) {
        if (pref_blocks[j + 1] - pref_blocks[i] < kMinSegBlocks) continue;
        double cost = 0.0;
        for (size_t co = 0; co < num_co; ++co) {
          const uint32_t* pi = &pref[(co * (K + 1) + i) * kQfBins];
          const uint32_t* pj = &pref[(co * (K + 1) + j + 1) * kQfBins];
          uint32_t n_nz = 0;
          int act_nz = 0;
          double sum_nz = 0.0;
          for (size_t b = 0; b < 8; ++b) {
            const uint32_t cnt = pj[b] - pi[b];
            if (cnt > 0) {
              n_nz += cnt;
              act_nz++;
              sum_nz += fast_nlog2n(cnt);
            }
          }
          if (act_nz > 1) {
            cost += (fast_nlog2n(n_nz) - sum_nz) +
                    kQfNullBias * static_cast<double>(act_nz - 1);
          }
          uint32_t n_ac = 0;
          int act_ac = 0;
          double sum_ac = 0.0;
          for (size_t b = 8; b < 16; ++b) {
            const uint32_t cnt = pj[b] - pi[b];
            if (cnt > 0) {
              n_ac += cnt;
              act_ac++;
              sum_ac += fast_nlog2n(cnt);
            }
          }
          if (act_ac > 1) {
            cost += (fast_nlog2n(n_ac) - sum_ac) +
                    kQfNullBias * static_cast<double>(act_ac - 1);
          }
        }
        seg_cost[i * K + j] = cost;
      }
    }

    const size_t max_s = std::min(max_q_segs, K);
    std::vector<double> dp((max_s + 1) * K, kInf);
    std::vector<int32_t> parent_split((max_s + 1) * K, -1);
    for (size_t j = 0; j < K; ++j) {
      dp[1 * K + j] = seg_cost[0 * K + j];
    }
    for (size_t s = 2; s <= max_s; ++s) {
      for (size_t j = s - 1; j < K; ++j) {
        double best_v = kInf;
        int32_t best_m = -1;
        for (size_t m = s - 2; m < j; ++m) {
          const double prev = dp[(s - 1) * K + m];
          const double cur = seg_cost[(m + 1) * K + j];
          if (prev < kInf * 0.5 && cur < kInf * 0.5 && prev + cur < best_v) {
            best_v = prev + cur;
            best_m = static_cast<int32_t>(m);
          }
        }
        dp[s * K + j] = best_v;
        parent_split[s * K + j] = best_m;
      }
    }

    constexpr double kMinBitsPerQfSplit = 250.0;
    size_t best_s = 1;
    double cur_best_cost = dp[1 * K + (K - 1)];
    for (size_t s = 2; s <= max_s; ++s) {
      const double cand = dp[s * K + (K - 1)];
      if (cand < kInf * 0.5 && cur_best_cost - cand > kMinBitsPerQfSplit) {
        best_s = s;
        cur_best_cost = cand;
      } else {
        break;
      }
    }
    if (best_s > 1) {
      std::vector<uint32_t> rev_splits;
      int32_t cur_j = static_cast<int32_t>(K - 1);
      for (size_t s = best_s; s >= 2; --s) {
        const int32_t m = parent_split[s * K + cur_j];
        if (m < 0) break;
        const uint32_t thr = std::min<uint32_t>(255, active_qfs[m]);
        rev_splits.push_back(thr);
        cur_j = m;
      }
      for (auto it = rev_splits.rbegin(); it != rev_splits.rend(); ++it) {
        if (block_ctx_map->qf_thresholds.empty() ||
            *it > block_ctx_map->qf_thresholds.back()) {
          block_ctx_map->qf_thresholds.push_back(*it);
        }
      }
    }
  }

  // Step 2: Collect exact 495-fine-context token histograms per active cell.
  constexpr size_t kNumFineCtx = kNonZeroBuckets + kZeroDensityContextCount;
  constexpr size_t kSymBins = 48;
  const size_t Q = block_ctx_map->qf_thresholds.size() + 1;
  const size_t M = 3 * kNumOrders * Q * num_dc_ctxs;

  struct CellCluster {
    uint8_t c_slice = 0;
    int32_t rep_ord = 0;
    int32_t rep_qf = 0;
    bool alive = true;
    uint32_t total_tokens = 0;
    std::vector<uint32_t> counts;
    std::array<uint32_t, kNumFineCtx> fine_tot{};
    std::array<double, kNumFineCtx> fine_ent{};
  };

  std::vector<int32_t> cell_to_active(M, -1);
  std::vector<CellCluster> clusters;
  clusters.reserve(64);

  JXL_ASSIGN_OR_RETURN(
      Image3I tmp_num_nzeroes,
      Image3I::Create(memory_manager, kGroupDimInBlocks, kGroupDimInBlocks));
  const size_t nzeros_stride = tmp_num_nzeroes.PixelsPerRow();
  const HybridUintConfig uint_config;

  for (size_t group_index = 0; group_index < frame_dim.num_groups;
       ++group_index) {
    const Rect grect = frame_dim.BlockGroupRect(group_index);
    const int32_t* JXL_RESTRICT ac_rows[3] = {
        coeffs.PlaneRow(0, group_index, 0).ptr32,
        coeffs.PlaneRow(1, group_index, 0).ptr32,
        coeffs.PlaneRow(2, group_index, 0).ptr32,
    };
    size_t offset[3] = {};
    for (size_t by = 0; by < grect.ysize(); ++by) {
      const size_t sby[3] = {by >> cs.VShift(0), by >> cs.VShift(1),
                             by >> cs.VShift(2)};
      int32_t* JXL_RESTRICT row_nzeros[3] = {
          tmp_num_nzeroes.PlaneRow(0, sby[0]),
          tmp_num_nzeroes.PlaneRow(1, sby[1]),
          tmp_num_nzeroes.PlaneRow(2, sby[2]),
      };
      const int32_t* JXL_RESTRICT row_nzeros_top[3] = {
          sby[0] == 0 ? nullptr : tmp_num_nzeroes.ConstPlaneRow(0, sby[0] - 1),
          sby[1] == 0 ? nullptr : tmp_num_nzeroes.ConstPlaneRow(1, sby[1] - 1),
          sby[2] == 0 ? nullptr : tmp_num_nzeroes.ConstPlaneRow(2, sby[2] - 1),
      };
      const uint8_t* JXL_RESTRICT row_qdc =
          qdc.ConstRow(grect.y0() + by) + grect.x0();
      const int32_t* JXL_RESTRICT row_qf = grect.ConstRow(rqf, by);
      AcStrategyRow acs_row = ac_strategy.ConstRow(grect, by);
      for (size_t bx = 0; bx < grect.xsize(); ++bx) {
        AcStrategy acs = acs_row[bx];
        if (!acs.IsFirstBlock()) continue;
        const size_t sbx[3] = {bx >> cs.HShift(0), bx >> cs.HShift(1),
                               bx >> cs.HShift(2)};
        const size_t cov_x = acs.covered_blocks_x();
        const size_t cov_y = acs.covered_blocks_y();
        const size_t covered_blocks = cov_x * cov_y;
        const size_t log2_covered_blocks =
            Num0BitsBelowLS1Bit_Nonzero(covered_blocks);
        const size_t size = covered_blocks * kDCTBlockSize;
        const int ord = kStrategyOrder[acs.RawStrategy()];

        for (int c : {1, 0, 2}) {
          if ((sbx[c] << cs.HShift(c)) != bx ||
              (sby[c] << cs.VShift(c)) != by) {
            continue;
          }
          const int32_t* JXL_RESTRICT block = ac_rows[c] + offset[c];
          const coeff_order_t* JXL_RESTRICT order =
              &coeff_orders[CoeffOrderOffset(ord, c)];

          int32_t nzeros = 0;
          for (size_t k = covered_blocks; k < size; ++k) {
            nzeros += (block[order[k]] != 0);
          }
          const int32_t shifted_nzeros = static_cast<int32_t>(
              (nzeros + covered_blocks - 1) >> log2_covered_blocks);
          for (size_t sy = 0; sy < cov_y; ++sy) {
            for (size_t sx = 0; sx < cov_x; ++sx) {
              row_nzeros[c][sbx[c] + sx + sy * nzeros_stride] = shifted_nzeros;
            }
          }

          const int32_t predicted_nzeros = PredictFromTopAndLeft(
              row_nzeros_top[c], row_nzeros[c], sbx[c], 32);
          const uint32_t clamped_pred_nz =
              std::min<uint32_t>(64, static_cast<uint32_t>(predicted_nzeros));
          const uint32_t nzero_bucket =
              (clamped_pred_nz < 8) ? clamped_pred_nz
                                    : (4 + clamped_pred_nz / 2);

          const uint32_t qf_val =
              static_cast<uint32_t>(std::max<int32_t>(1, row_qf[sbx[c]]));
          size_t qf_idx = 0;
          for (uint32_t t : block_ctx_map->qf_thresholds) {
            if (qf_val > t) qf_idx++;
          }
          const size_t dc_idx =
              (num_dc_ctxs > 1)
                  ? std::min<size_t>(row_qdc[bx], num_dc_ctxs - 1)
                  : 0;
          const size_t c_slice = (c < 2) ? (c ^ 1) : 2;
          const size_t cell_idx =
              ((c_slice * kNumOrders + ord) * Q + qf_idx) * num_dc_ctxs +
              dc_idx;

          int32_t act_id = cell_to_active[cell_idx];
          if (act_id < 0) {
            act_id = static_cast<int32_t>(clusters.size());
            cell_to_active[cell_idx] = act_id;
            clusters.emplace_back();
            auto& cl = clusters.back();
            cl.c_slice = static_cast<uint8_t>(c_slice);
            cl.rep_ord = ord;
            cl.rep_qf = static_cast<int32_t>(qf_idx);
            cl.counts.assign(kNumFineCtx * kSymBins, 0);
          }
          auto& cl = clusters[act_id];

          uint32_t tok_nz = 0;
          uint32_t nbits = 0;
          uint32_t bits = 0;
          uint_config.Encode(static_cast<uint32_t>(nzeros), &tok_nz, &nbits,
                             &bits);
          tok_nz = std::min<uint32_t>(tok_nz, kSymBins - 1);
          cl.counts[nzero_bucket * kSymBins + tok_nz]++;
          cl.fine_tot[nzero_bucket]++;
          cl.total_tokens++;

          size_t prev = (nzeros > static_cast<ptrdiff_t>(size / 16) ? 0 : 1);
          int32_t rem_nz = nzeros;
          for (size_t k = covered_blocks; k < size && rem_nz != 0; ++k) {
            const int32_t coeff = block[order[k]];
            const size_t zd_bucket =
                kNonZeroBuckets + ZeroDensityContext(rem_nz, k, covered_blocks,
                                                     log2_covered_blocks, prev);
            const uint32_t u_coeff = PackSigned(coeff);
            uint32_t tok_ac = 0;
            uint_config.Encode(u_coeff, &tok_ac, &nbits, &bits);
            tok_ac = std::min<uint32_t>(tok_ac, kSymBins - 1);
            cl.counts[zd_bucket * kSymBins + tok_ac]++;
            cl.fine_tot[zd_bucket]++;
            cl.total_tokens++;
            prev = (coeff != 0) ? 1 : 0;
            rem_nz -= static_cast<int32_t>(prev);
          }
          offset[c] += size;
        }
      }
    }
  }

  const size_t N_active = clusters.size();
  if (N_active == 0) {
    return true;
  }

  // Step 3: Compute fine-context entropies and run greedy agglomerative
  // Jensen-Shannon clustering (keeping Luma and Chroma disjoint).
  const auto compute_fine_ent = [&](CellCluster& cl, size_t f) {
    const uint32_t n = cl.fine_tot[f];
    if (n <= 1) {
      cl.fine_ent[f] = 0.0;
      return;
    }
    const uint32_t* cnts = &cl.counts[f * kSymBins];
    double sum_c = 0.0;
    for (size_t s = 0; s < kSymBins; ++s) {
      if (cnts[s] > 1) sum_c += fast_nlog2n(cnts[s]);
    }
    cl.fine_ent[f] = fast_nlog2n(n) - sum_c;
  };
  for (size_t a = 0; a < N_active; ++a) {
    for (size_t f = 0; f < kNumFineCtx; ++f) {
      if (clusters[a].fine_tot[f] > 0) compute_fine_ent(clusters[a], f);
    }
  }

  constexpr double kMergeNullBias = 0.70;
  constexpr double kMergeLambda = 1000.0;
  constexpr double kInfCost = 1e30;

  const auto merge_cost = [&](size_t a, size_t b) -> double {
    const auto& ca = clusters[a];
    const auto& cb = clusters[b];
    if ((ca.c_slice == 0) != (cb.c_slice == 0)) return kInfCost;
    double total_de = 0.0;
    for (size_t f = 0; f < kNumFineCtx; ++f) {
      const uint32_t na = ca.fine_tot[f];
      const uint32_t nb = cb.fine_tot[f];
      if (na == 0 || nb == 0) continue;
      const uint32_t* ha = &ca.counts[f * kSymBins];
      const uint32_t* hb = &cb.counts[f * kSymBins];
      double sum_ab = 0.0;
      int overlap = 0;
      for (size_t s = 0; s < kSymBins; ++s) {
        const uint32_t cab = ha[s] + hb[s];
        if (cab > 0) {
          overlap++;
          if (cab > 1) sum_ab += fast_nlog2n(cab);
        }
      }
      if (overlap <= 1) continue;
      const double ent_ab = fast_nlog2n(na + nb) - sum_ab;
      const double raw_de = ent_ab - ca.fine_ent[f] - cb.fine_ent[f];
      const double bias = kMergeNullBias * static_cast<double>(overlap - 1);
      if (raw_de > bias) {
        total_de += (raw_de - bias);
      }
    }
    const double tie =
        1e-3 * (4.0 * static_cast<double>(ca.c_slice != cb.c_slice) +
                static_cast<double>(std::abs(ca.rep_ord - cb.rep_ord)) +
                0.25 * static_cast<double>(std::abs(ca.rep_qf - cb.rep_qf)));
    return total_de + tie;
  };

  std::vector<double> dist_matrix(N_active * N_active, kInfCost);
  for (size_t a = 0; a < N_active; ++a) {
    for (size_t b = a + 1; b < N_active; ++b) {
      const double d = merge_cost(a, b);
      dist_matrix[a * N_active + b] = d;
      dist_matrix[b * N_active + a] = d;
    }
  }

  std::vector<size_t> parent(N_active);
  std::iota(parent.begin(), parent.end(), 0);
  size_t luma_alive = 0;
  size_t chroma_alive = 0;
  for (size_t a = 0; a < N_active; ++a) {
    if (clusters[a].c_slice == 0) {
      luma_alive++;
    } else {
      chroma_alive++;
    }
  }

  while (luma_alive > 1 || chroma_alive > 1) {
    const bool force_luma = (luma_alive > max_luma_ctxs) ||
                            (luma_alive + chroma_alive > 16 &&
                             luma_alive > 1);
    const bool force_chroma = (chroma_alive > max_chroma_ctxs) ||
                              (luma_alive + chroma_alive > 16 &&
                               chroma_alive > 1);
    double best_d = kInfCost;
    int32_t best_a = -1;
    int32_t best_b = -1;
    for (size_t a = 0; a < N_active; ++a) {
      if (!clusters[a].alive) continue;
      const bool is_luma = (clusters[a].c_slice == 0);
      if (is_luma && luma_alive <= 1) continue;
      if (!is_luma && chroma_alive <= 1) continue;
      for (size_t b = a + 1; b < N_active; ++b) {
        if (!clusters[b].alive) continue;
        if ((clusters[b].c_slice == 0) != is_luma) continue;
        const double d = dist_matrix[a * N_active + b];
        if (!force_luma && !force_chroma && d >= kMergeLambda) continue;
        if (is_luma && !force_luma && d >= kMergeLambda) continue;
        if (!is_luma && !force_chroma && d >= kMergeLambda) continue;
        if (d < best_d) {
          best_d = d;
          best_a = static_cast<int32_t>(a);
          best_b = static_cast<int32_t>(b);
        }
      }
    }
    if (best_a < 0 || best_d >= kInfCost * 0.5) break;

    auto& ca = clusters[best_a];
    auto& cb = clusters[best_b];
    for (size_t f = 0; f < kNumFineCtx; ++f) {
      if (cb.fine_tot[f] == 0) continue;
      uint32_t* ha = &ca.counts[f * kSymBins];
      const uint32_t* hb = &cb.counts[f * kSymBins];
      for (size_t s = 0; s < kSymBins; ++s) {
        ha[s] += hb[s];
      }
      ca.fine_tot[f] += cb.fine_tot[f];
      compute_fine_ent(ca, f);
    }
    ca.total_tokens += cb.total_tokens;
    cb.alive = false;
    parent[best_b] = static_cast<size_t>(best_a);
    if (ca.c_slice == 0) {
      luma_alive--;
    } else {
      chroma_alive--;
    }

    for (size_t x = 0; x < N_active; ++x) {
      if (x == static_cast<size_t>(best_a) || !clusters[x].alive) continue;
      const double d = merge_cost(best_a, x);
      dist_matrix[best_a * N_active + x] = d;
      dist_matrix[x * N_active + best_a] = d;
    }
  }

  for (size_t a = 0; a < N_active; ++a) {
    while (parent[parent[a]] != parent[a]) {
      parent[a] = parent[parent[a]];
    }
  }

  // Step 4: Fill ctx_map, propagate empty cells into contiguous MTF runs,
  // and relabel cluster IDs in first-appearance order.
  std::vector<int32_t> raw_ctx(M, -1);
  for (size_t idx = 0; idx < M; ++idx) {
    const int32_t act = cell_to_active[idx];
    if (act >= 0) {
      raw_ctx[idx] = static_cast<int32_t>(parent[act]);
    }
  }

  const size_t ord_span = Q * num_dc_ctxs;
  const size_t slice_span = kNumOrders * ord_span;
  for (size_t cs_idx = 0; cs_idx < 3; ++cs_idx) {
    const size_t cs_base = cs_idx * slice_span;
    for (size_t ord = 0; ord < kNumOrders; ++ord) {
      const size_t ob = cs_base + ord * ord_span;
      for (size_t i = 1; i < ord_span; ++i) {
        if (raw_ctx[ob + i] < 0 && raw_ctx[ob + i - 1] >= 0) {
          raw_ctx[ob + i] = raw_ctx[ob + i - 1];
        }
      }
      for (size_t i = ord_span - 1; i > 0; --i) {
        if (raw_ctx[ob + i - 1] < 0 && raw_ctx[ob + i] >= 0) {
          raw_ctx[ob + i - 1] = raw_ctx[ob + i];
        }
      }
    }
    for (size_t i = 1; i < slice_span; ++i) {
      if (raw_ctx[cs_base + i] < 0 && raw_ctx[cs_base + i - 1] >= 0) {
        raw_ctx[cs_base + i] = raw_ctx[cs_base + i - 1];
      }
    }
    for (size_t i = slice_span - 1; i > 0; --i) {
      if (raw_ctx[cs_base + i - 1] < 0 && raw_ctx[cs_base + i] >= 0) {
        raw_ctx[cs_base + i - 1] = raw_ctx[cs_base + i];
      }
    }
  }
  for (size_t i = 1; i < M; ++i) {
    if (raw_ctx[i] < 0 && raw_ctx[i - 1] >= 0) raw_ctx[i] = raw_ctx[i - 1];
  }
  for (size_t i = M - 1; i > 0; --i) {
    if (raw_ctx[i - 1] < 0 && raw_ctx[i] >= 0) raw_ctx[i - 1] = raw_ctx[i];
  }

  std::vector<int32_t> root_to_id(N_active, -1);
  uint8_t next_id = 0;
  block_ctx_map->ctx_map.resize(M);
  for (size_t i = 0; i < M; ++i) {
    const int32_t r = std::max<int32_t>(0, raw_ctx[i]);
    if (root_to_id[r] < 0) {
      root_to_id[r] = next_id++;
    }
    block_ctx_map->ctx_map[i] = static_cast<uint8_t>(root_to_id[r]);
  }
  block_ctx_map->num_ctxs = std::max<size_t>(1, next_id);
  JXL_ENSURE(block_ctx_map->num_ctxs <= 16);
  return true;
}

namespace {

Status FindBestDequantMatrices(JxlMemoryManager* memory_manager,
                               const CompressParams& cparams,
                               const FrameHeader& frame_header,
                               const FrameDimensions& frame_dim,
                               bool enable_rdo,
                               const ImageF& initial_quant_field,
                               const ImageF (*band_qf)[4],
                               ModularFrameEncoder* modular_frame_encoder,
                               DequantMatrices* dequant_matrices) {
  // TODO(veluca): quant matrices for no-gaborish.
  // TODO(veluca): heuristics for in-bitstream quant tables.
  *dequant_matrices = DequantMatrices();
  if (cparams.max_error_mode || cparams.disable_perceptual_optimizations) {
    constexpr float kMSEWeights[3] = {0.001, 0.001, 0.001};
    const float* wp = cparams.disable_perceptual_optimizations
                          ? kMSEWeights
                          : cparams.max_error;
    // Set numerators of all quantization matrices to constant values.
    float weights[3][1] = {{1.0f / wp[0]}, {1.0f / wp[1]}, {1.0f / wp[2]}};
    DctQuantWeightParams dct_params(weights);
    std::vector<QuantEncoding> encodings(kNumQuantTables,
                                         QuantEncoding::DCT(dct_params));
    JXL_RETURN_IF_ERROR(DequantMatricesSetCustom(dequant_matrices, encodings,
                                                 modular_frame_encoder));
    float dc_weights[3] = {1.0f / wp[0], 1.0f / wp[1], 1.0f / wp[2]};
    JXL_RETURN_IF_ERROR(DequantMatricesSetCustomDC(
        memory_manager, dequant_matrices, dc_weights));
  } else if (enable_rdo) {
    (void)frame_dim;
    // 8 VarDCT kQuantModeDCT tables ordered by log-block-scale b:
    //   idx 0: DCT8     (b=0.0, a=0.0)
    //   idx 1: DCT8X16  (b=0.5, a=0.5)
    //   idx 2: DCT16X16 (b=1.0, a=0.0)
    //   idx 3: DCT8X32  (b=1.0, a=1.0)
    //   idx 4: DCT16X32 (b=1.5, a=0.5)
    //   idx 5: DCT32X32 (b=2.0, a=0.0)
    //   idx 6: DCT32X64 (b=2.5, a=0.5)
    //   idx 7: DCT64X64 (b=3.0, a=0.0)
    struct DctTableSpec {
      QuantTable table;
      float b;
      float a;
      size_t num_bands;
    };
    static constexpr DctTableSpec kDctTables[8] = {
        {QuantTable::DCT, 0.0f, 0.0f, 6},
        {QuantTable::DCT8X16, 0.5f, 0.5f, 7},
        {QuantTable::DCT16X16, 1.0f, 0.0f, 7},
        {QuantTable::DCT8X32, 1.0f, 1.0f, 8},
        {QuantTable::DCT16X32, 1.5f, 0.5f, 8},
        {QuantTable::DCT32X32, 2.0f, 0.0f, 8},
        {QuantTable::DCT32X64, 2.5f, 0.5f, 8},
        {QuantTable::DCT64X64, 3.0f, 0.0f, 8},
    };

    struct ChannelQMParams {
      float W0;
      float s1;
      float s2;
      float rho;
      float D0;
      float Db;
      float k0;
      float kb;
      float gamma_d;
      float beta_blk;
      float eta_d;
      float a_mid0;
      float a_mid_d;
      float w0_dslope;
      float d0_dslope;
    };
    // Principled from-scratch parametric DCT quantization model for X, Y, B
    // (including smooth log(d / 1.85) distance slopes for steepness & chroma).
    static constexpr ChannelQMParams kChannelParams[3] = {
        // c = 0 (X):
        {4380.35547f, 0.615591586f, -0.185f, 0.055f, 2.09565282f, 0.885f, 1.10f,
         1.53f, 0.010f, 0.003f, 0.012f, 0.022f, 0.004f, -0.0135177998f,
         0.0230346546f},
        // c = 1 (Y):
        {808.807312f, 0.653820932f, 0.966f, 0.242f, 1.7416153f, 1.269f, 0.14f,
         0.67f, 0.000f, 0.004f, 0.007f, 0.029f, 0.005f, 0.0f, 0.0208702553f},
        // c = 2 (B): convex 4-metric compromise — preserve LF/MF blue-yellow
        // contrast for WD_s1 & SSIM2 while retaining S-cone HF roll-off so bits
        // are not wasted on invisible HF blue grain under BA3 & CVVDP.
        {290.563446f, 0.715576172f, 0.520f, 0.115f, 2.04612994f, 1.295f, 0.46f,
         1.02f, 0.011f, 0.004f, 0.012f, 0.026f, 0.005f, -0.000394175004f,
         0.0368712395f},
    };

    const float d = Clamp1(cparams.butteraugli_distance, 0.35f, 15.0f);
    const float t_d = Clamp1((d - 0.5f) / 2.0f, 0.0f, 1.0f);
    const float s_d = Clamp1(std::log(d * (1.0f / 1.85f)), -1.45f, 1.85f);
    constexpr float kU0 = 0.10f;

    auto inv_mult_step = [](float r) -> float {
      return (r >= 1.0f) ? (r - 1.0f) : (1.0f - 1.0f / r);
    };
    float delta_hf[3] = {};
    float delta_mf[3] = {};
    float delta_chr[3] = {};
    if (band_qf != nullptr && band_qf[1][0].xsize() > 0 &&
        band_qf[1][0].ysize() > 0 &&
        frame_dim.xsize_blocks * frame_dim.ysize_blocks > 1) {
      const float hi_d_taper = Clamp1((d - 2.5f) / 7.5f, 0.0f, 1.0f);
      const float w_bd =
          0.52f * (1.0f - 0.30f * hi_d_taper) *
          Clamp1((d - 0.35f) * 2.0f, 0.0f, 1.0f);
      if (w_bd > 1e-3f) {
        const float inv_w_bd = 1.0f / std::max(w_bd, 0.15f);
        const float ln_r03_flat =
            std::log((1.0f + w_bd * 0.0200f) / (1.0f + w_bd * 0.0710f));
        const float ln_r12_flat =
            std::log((1.0f + w_bd * 0.0404f) / (1.0f + w_bd * 0.0302f));
        double w_sum = 0.0;
        double act_sum = 0.0;
        double sum_hf[3] = {};
        double sum_mf[3] = {};
        double sum_chr[3] = {};
        const size_t total_blocks =
            frame_dim.xsize_blocks * frame_dim.ysize_blocks;
        for (size_t by = 0; by < frame_dim.ysize_blocks; ++by) {
          const float* JXL_RESTRICT q0_row = initial_quant_field.ConstRow(by);
          const float* JXL_RESTRICT bq_row[3][4];
          for (size_t c = 0; c < 3; ++c) {
            for (size_t s = 0; s < 4; ++s) {
              bq_row[c][s] = band_qf[c][s].ConstRow(by);
            }
          }
          for (size_t bx = 0; bx < frame_dim.xsize_blocks; ++bx) {
            uint32_t act_bits = 0;
            memcpy(&act_bits, &bq_row[1][0][bx], sizeof(act_bits));
            const int32_t activity =
                static_cast<int32_t>((act_bits >> 1) & 15u);
            const float act_w =
                std::min(1.0f, static_cast<float>(activity) * (1.0f / 3.0f));
            act_sum += act_w;
            const float q_base = std::max(q0_row[bx], 1e-4f);
            const float bq12 = std::max(bq_row[1][2][bx], 1e-4f);
            const float w_blk =
                0.05f + 0.65f * act_w +
                0.30f * Clamp1(std::log((1.18f * q_base) / bq12), 0.0f, 1.0f);
            w_sum += w_blk;
            for (size_t c = 0; c < 3; ++c) {
              const float b0 = std::max(bq_row[c][0][bx], 1e-4f);
              const float b1 = std::max(bq_row[c][1][bx], 1e-4f);
              const float b2 = std::max(bq_row[c][2][bx], 1e-4f);
              const float b3 = std::max(bq_row[c][3][bx], 1e-4f);
              sum_hf[c] += w_blk * (std::log(b0 / b3) - ln_r03_flat) * inv_w_bd;
              sum_mf[c] += w_blk * (std::log(b1 / b2) - ln_r12_flat) * inv_w_bd;
              if (c != 1) {
                sum_chr[c] += w_blk * std::log(b2 / bq12) * inv_w_bd;
              }
            }
          }
        }
        if (w_sum > 1e-6 && total_blocks > 0) {
          static constexpr float kHfRef[3] = {-0.085f, -0.076f, -0.080f};
          static constexpr float kMfRef[3] = {-0.036f, -0.045f, -0.034f};
          static constexpr float kChrRef[3] = {0.008f, 0.0f, -0.002f};
          const float act_frac =
              static_cast<float>(act_sum) / static_cast<float>(total_blocks);
          const float act_gate =
              Clamp1((act_frac - 0.04f) / 0.22f, 0.0f, 1.0f);
          for (size_t c = 0; c < 3; ++c) {
            const float hf_t = static_cast<float>(sum_hf[c] / w_sum);
            const float mf_t = static_cast<float>(sum_mf[c] / w_sum);
            const float chr_t = static_cast<float>(sum_chr[c] / w_sum);
            delta_hf[c] =
                act_gate *
                (-0.010f + Clamp1(hf_t - kHfRef[c], -0.035f, 0.0f));
            const float raw_mf =
                act_gate *
                Clamp1((mf_t - kMfRef[c]) - 0.40f * (hf_t - kHfRef[c]), -0.004f,
                       0.009f);
            delta_mf[c] = std::min(raw_mf, -0.38f * delta_hf[c]);
            delta_chr[c] =
                (c != 1)
                    ? (act_gate * Clamp1(chr_t - kChrRef[c], -0.030f, 0.0f))
                    : 0.0f;
          }
        }
      }
    }

    const size_t frame_blocks = frame_dim.xsize_blocks * frame_dim.ysize_blocks;
    const size_t max_bands =
        (frame_blocks < 1024 || d >= 5.0f)
            ? 4
            : ((frame_blocks < 2500 || d >= 2.5f) ? 5 : 8);
    std::vector<QuantEncoding> encodings(kNumQuantTables,
                                         QuantEncoding::Library<0>());
    for (size_t idx = 0; idx < 8; ++idx) {
      const DctTableSpec& spec = kDctTables[idx];
      const size_t out_bands = std::min(spec.num_bands, max_bands);
      const float b = spec.b;
      const float a = spec.a;
      const float tx_decay = std::exp(-0.18f * b);
      DctQuantWeightParams new_params;
      new_params.num_distance_bands = out_bands;

      for (size_t c = 0; c < 3; ++c) {
        const ChannelQMParams& p = kChannelParams[c];
        const float w0_eff =
            (p.w0_dslope != 0.0f) ? (p.W0 * std::exp(p.w0_dslope * s_d)) : p.W0;
        const float d0_eff = p.D0 * std::exp(p.d0_dslope * s_d);
        const float db_eff = p.Db * std::exp(0.0447954014f * s_d);
        const float ln_w_u0 =
            std::log(w0_eff) + p.gamma_d * t_d +
            (p.s1 * b + p.s2 * (b / (0.5f + b)) -
             p.rho * (a / (0.20f + b))) *
                (1.0f - p.beta_blk * t_d);
        const float D =
            (d0_eff + db_eff * std::log(1.0f + b)) * (1.0f + p.eta_d * t_d);
        float kap = p.k0 - p.kb * (b / (0.25f + b));
        kap = Clamp1(kap + (-0.0169299729f) * s_d, -1.5f, 1.8f);
        const float a_mid = (p.a_mid0 - p.a_mid_d * t_d) * tx_decay;

        auto eval_ln_w = [&](float u) -> float {
          const float v = (u - kU0) / (1.0f - kU0);
          const float phi = (1.0f - kap) * v + kap * v * std::abs(v);
          const float s = v - 0.34f;
          const float hf_dec =
              (c == 1 || s <= 0.0f) ? 0.45f : ((c == 0) ? 0.90f : 1.10f);
          const float psi = a_mid * s * (1.15f - hf_dec * t_d * v);
          return ln_w_u0 - D * phi + psi;
        };

        float target_w[DctQuantWeightParams::kMaxDistanceBands] = {};
        const float u1 = 1.0f / static_cast<float>(out_bands - 1);
        const float ln_w_1 = eval_ln_w(u1);
        const float ln_w_0 =
            (u1 > kU0 + 1e-4f)
                ? (ln_w_u0 - (kU0 / (u1 - kU0)) * (ln_w_1 - ln_w_u0))
                : eval_ln_w(0.0f);
        target_w[0] = std::exp(ln_w_0);
        target_w[1] = std::exp(ln_w_1);
        const float d_gate = Clamp1((d - 1.05f) / 0.45f, 0.0f, 1.0f) *
                             (1.0f - 0.35f * Clamp1(d - 1.50f, 0.0f, 1.0f));
        const float hf_adapt =
            d_gate * (0.22f * delta_hf[c] + 0.14f * delta_chr[c]);
        const float mf_adapt =
            d_gate * 0.22f * delta_mf[c] * std::exp(-0.55f * b);
        for (size_t k = 2; k < out_bands; ++k) {
          const float u =
              static_cast<float>(k) / static_cast<float>(out_bands - 1);
          const float xi = (u - u1) / (1.0f - u1);
          const float mod =
              hf_adapt * xi * xi + mf_adapt * 4.0f * xi * (1.0f - xi);
          target_w[k] = std::exp(eval_ln_w(u) + mod);
        }

        new_params.distance_bands[c][0] = target_w[0];
        for (size_t k = 1; k < out_bands; ++k) {
          new_params.distance_bands[c][k] =
              inv_mult_step(target_w[k] / target_w[k - 1]);
        }
      }
      encodings[static_cast<size_t>(spec.table)] =
          QuantEncoding::DCT(new_params);
    }

    JXL_RETURN_IF_ERROR(DequantMatricesSetCustom(dequant_matrices, encodings,
                                                 modular_frame_encoder));
    if (frame_blocks >= 256 &&
        frame_header.color_transform == ColorTransform::kXYB &&
        frame_header.chroma_subsampling.Is444()) {
      // Align XYB DC channel precision with the 4-metric convex RDO optimum:
      // refine Y and B DC precision to preserve multi-scale pyramid structure
      // (SSIM2, CVVDP, WD_s1) and LLF reconstruction while keeping X DC at the
      // WD_s1-safe operating point.
      float x_dc_scale =
          0.757971644f - 0.143665373f * Clamp1((d - 1.0f) / 3.0f, 0.0f, 1.0f);
      float y_dc_scale =
          1.17833197f - 0.14812851f * Clamp1((d - 1.0f) / 4.5f, 0.0f, 1.0f);
      float b_dc_scale =
          y_dc_scale *
          (0.957945466f + 0.03f * Clamp1((d - 1.5f) / 4.0f, 0.0f, 1.0f));
      const float all_m = std::exp(0.00649557495f * s_d);
      x_dc_scale *= all_m * std::exp(-0.0357682817f * s_d);
      y_dc_scale *= all_m;
      b_dc_scale *= all_m * std::exp(-0.00723599782f * s_d);
      const float custom_inv_dc[3] = {
          kInvDCQuant[0] * x_dc_scale,
          kInvDCQuant[1] * y_dc_scale,
          kInvDCQuant[2] * b_dc_scale,
      };
      JXL_RETURN_IF_ERROR(DequantMatricesSetCustomDC(
          memory_manager, dequant_matrices, custom_inv_dc));
    }
  }
  return true;
}

void StoreMin2(const float v, float& min1, float& min2) {
  if (v < min2) {
    if (v < min1) {
      min2 = min1;
      min1 = v;
    } else {
      min2 = v;
    }
  }
}

void CreateMask(const ImageF& image, ImageF& mask) {
  for (size_t y = 0; y < image.ysize(); y++) {
    const auto* row_n = y > 0 ? image.Row(y - 1) : image.Row(y);
    const auto* row_in = image.Row(y);
    const auto* row_s = y + 1 < image.ysize() ? image.Row(y + 1) : image.Row(y);
    auto* row_out = mask.Row(y);
    for (size_t x = 0; x < image.xsize(); x++) {
      // Center, west, east, north, south values and their absolute difference
      float c = row_in[x];
      float w = x > 0 ? row_in[x - 1] : row_in[x];
      float e = x + 1 < image.xsize() ? row_in[x + 1] : row_in[x];
      float n = row_n[x];
      float s = row_s[x];
      float dw = std::abs(c - w);
      float de = std::abs(c - e);
      float dn = std::abs(c - n);
      float ds = std::abs(c - s);
      float min = std::numeric_limits<float>::max();
      float min2 = std::numeric_limits<float>::max();
      StoreMin2(dw, min, min2);
      StoreMin2(de, min, min2);
      StoreMin2(dn, min, min2);
      StoreMin2(ds, min, min2);
      row_out[x] = min2;
    }
  }
}

// Downsamples the image by a factor of 2 with a kernel that's sharper than
// the standard 2x2 box kernel used by DownsampleImage.
// The kernel is optimized against the result of the 2x2 upsampling kernel used
// by the decoder. Ringing is slightly reduced by clamping the values of the
// resulting pixels within certain bounds of a small region in the original
// image.
Status DownsampleImage2_Sharper(const ImageF& input, ImageF* output) {
  const int64_t kernelx = 12;
  const int64_t kernely = 12;
  JxlMemoryManager* memory_manager = input.memory_manager();

  static const float kernel[144] = {
      -0.000314256996835, -0.000314256996835, -0.000897597057705,
      -0.000562751488849, -0.000176807273646, 0.001864627368902,
      0.001864627368902,  -0.000176807273646, -0.000562751488849,
      -0.000897597057705, -0.000314256996835, -0.000314256996835,
      -0.000314256996835, -0.001527942804748, -0.000121760530512,
      0.000191123989093,  0.010193185932466,  0.058637519197110,
      0.058637519197110,  0.010193185932466,  0.000191123989093,
      -0.000121760530512, -0.001527942804748, -0.000314256996835,
      -0.000897597057705, -0.000121760530512, 0.000946363683751,
      0.007113577630288,  0.000437956841058,  -0.000372823835211,
      -0.000372823835211, 0.000437956841058,  0.007113577630288,
      0.000946363683751,  -0.000121760530512, -0.000897597057705,
      -0.000562751488849, 0.000191123989093,  0.007113577630288,
      0.044592622228814,  0.000222278879007,  -0.162864473015945,
      -0.162864473015945, 0.000222278879007,  0.044592622228814,
      0.007113577630288,  0.000191123989093,  -0.000562751488849,
      -0.000176807273646, 0.010193185932466,  0.000437956841058,
      0.000222278879007,  -0.000913092543974, -0.017071696107902,
      -0.017071696107902, -0.000913092543974, 0.000222278879007,
      0.000437956841058,  0.010193185932466,  -0.000176807273646,
      0.001864627368902,  0.058637519197110,  -0.000372823835211,
      -0.162864473015945, -0.017071696107902, 0.414660099370354,
      0.414660099370354,  -0.017071696107902, -0.162864473015945,
      -0.000372823835211, 0.058637519197110,  0.001864627368902,
      0.001864627368902,  0.058637519197110,  -0.000372823835211,
      -0.162864473015945, -0.017071696107902, 0.414660099370354,
      0.414660099370354,  -0.017071696107902, -0.162864473015945,
      -0.000372823835211, 0.058637519197110,  0.001864627368902,
      -0.000176807273646, 0.010193185932466,  0.000437956841058,
      0.000222278879007,  -0.000913092543974, -0.017071696107902,
      -0.017071696107902, -0.000913092543974, 0.000222278879007,
      0.000437956841058,  0.010193185932466,  -0.000176807273646,
      -0.000562751488849, 0.000191123989093,  0.007113577630288,
      0.044592622228814,  0.000222278879007,  -0.162864473015945,
      -0.162864473015945, 0.000222278879007,  0.044592622228814,
      0.007113577630288,  0.000191123989093,  -0.000562751488849,
      -0.000897597057705, -0.000121760530512, 0.000946363683751,
      0.007113577630288,  0.000437956841058,  -0.000372823835211,
      -0.000372823835211, 0.000437956841058,  0.007113577630288,
      0.000946363683751,  -0.000121760530512, -0.000897597057705,
      -0.000314256996835, -0.001527942804748, -0.000121760530512,
      0.000191123989093,  0.010193185932466,  0.058637519197110,
      0.058637519197110,  0.010193185932466,  0.000191123989093,
      -0.000121760530512, -0.001527942804748, -0.000314256996835,
      -0.000314256996835, -0.000314256996835, -0.000897597057705,
      -0.000562751488849, -0.000176807273646, 0.001864627368902,
      0.001864627368902,  -0.000176807273646, -0.000562751488849,
      -0.000897597057705, -0.000314256996835, -0.000314256996835};

  int64_t xsize = input.xsize();
  int64_t ysize = input.ysize();

  JXL_ASSIGN_OR_RETURN(ImageF box_downsample,
                       ImageF::Create(memory_manager, xsize, ysize));
  JXL_RETURN_IF_ERROR(CopyImageTo(input, &box_downsample));
  JXL_ASSIGN_OR_RETURN(box_downsample, DownsampleImage(box_downsample, 2));

  JXL_ASSIGN_OR_RETURN(ImageF mask,
                       ImageF::Create(memory_manager, box_downsample.xsize(),
                                      box_downsample.ysize()));
  CreateMask(box_downsample, mask);

  for (size_t y = 0; y < output->ysize(); y++) {
    float* row_out = output->Row(y);
    const float* row_in[kernely];
    const float* row_mask = mask.Row(y);
    // get the rows in the support
    for (size_t ky = 0; ky < kernely; ky++) {
      int64_t iy = y * 2 + ky - (kernely - 1) / 2;
      if (iy < 0) iy = 0;
      if (iy >= ysize) iy = ysize - 1;
      row_in[ky] = input.Row(iy);
    }

    for (size_t x = 0; x < output->xsize(); x++) {
      // get min and max values of the original image in the support
      float min = std::numeric_limits<float>::max();
      float max = std::numeric_limits<float>::min();
      // kernelx - R and kernely - R are the radius of a rectangular region in
      // which the values of a pixel are bounded to reduce ringing.
      static constexpr int64_t R = 5;
      for (int64_t ky = R; ky + R < kernely; ky++) {
        for (int64_t kx = R; kx + R < kernelx; kx++) {
          int64_t ix = x * 2 + kx - (kernelx - 1) / 2;
          if (ix < 0) ix = 0;
          if (ix >= xsize) ix = xsize - 1;
          min = std::min<float>(min, row_in[ky][ix]);
          max = std::max<float>(max, row_in[ky][ix]);
        }
      }

      float sum = 0;
      for (int64_t ky = 0; ky < kernely; ky++) {
        for (int64_t kx = 0; kx < kernelx; kx++) {
          int64_t ix = x * 2 + kx - (kernelx - 1) / 2;
          if (ix < 0) ix = 0;
          if (ix >= xsize) ix = xsize - 1;
          sum += row_in[ky][ix] * kernel[ky * kernelx + kx];
        }
      }

      row_out[x] = sum;

      // Clamp the pixel within the value  of a small area to prevent ringning.
      // The mask determines how much to clamp, clamp more to reduce more
      // ringing in smooth areas, clamp less in noisy areas to get more
      // sharpness. Higher mask_multiplier gives less clamping, so less
      // ringing reduction.
      const constexpr float mask_multiplier = 1;
      float a = row_mask[x] * mask_multiplier;
      float clip_min = min - a;
      float clip_max = max + a;
      if (row_out[x] < clip_min) {
        row_out[x] = clip_min;
      } else if (row_out[x] > clip_max) {
        row_out[x] = clip_max;
      }
    }
  }
  return true;
}

}  // namespace

Status DownsampleImage2_Sharper(Image3F* opsin) {
  // Allocate extra space to avoid a reallocation when padding.
  JxlMemoryManager* memory_manager = opsin->memory_manager();
  JXL_ASSIGN_OR_RETURN(
      Image3F downsampled,
      Image3F::Create(memory_manager, DivCeil(opsin->xsize(), 2) + kBlockDim,
                      DivCeil(opsin->ysize(), 2) + kBlockDim));
  JXL_RETURN_IF_ERROR(downsampled.ShrinkTo(downsampled.xsize() - kBlockDim,
                                           downsampled.ysize() - kBlockDim));

  for (size_t c = 0; c < 3; c++) {
    JXL_RETURN_IF_ERROR(
        DownsampleImage2_Sharper(opsin->Plane(c), &downsampled.Plane(c)));
  }
  *opsin = std::move(downsampled);
  return true;
}

namespace {

// The default upsampling kernels used by Upsampler in the decoder.
const constexpr int64_t kSize = 5;

const float kernel00[25] = {
    -0.01716200f, -0.03452303f, -0.04022174f, -0.02921014f, -0.00624645f,
    -0.03452303f, 0.14111091f,  0.28896755f,  0.00278718f,  -0.01610267f,
    -0.04022174f, 0.28896755f,  0.56661550f,  0.03777607f,  -0.01986694f,
    -0.02921014f, 0.00278718f,  0.03777607f,  -0.03144731f, -0.01185068f,
    -0.00624645f, -0.01610267f, -0.01986694f, -0.01185068f, -0.00213539f,
};
const float kernel01[25] = {
    -0.00624645f, -0.01610267f, -0.01986694f, -0.01185068f, -0.00213539f,
    -0.02921014f, 0.00278718f,  0.03777607f,  -0.03144731f, -0.01185068f,
    -0.04022174f, 0.28896755f,  0.56661550f,  0.03777607f,  -0.01986694f,
    -0.03452303f, 0.14111091f,  0.28896755f,  0.00278718f,  -0.01610267f,
    -0.01716200f, -0.03452303f, -0.04022174f, -0.02921014f, -0.00624645f,
};
const float kernel10[25] = {
    -0.00624645f, -0.02921014f, -0.04022174f, -0.03452303f, -0.01716200f,
    -0.01610267f, 0.00278718f,  0.28896755f,  0.14111091f,  -0.03452303f,
    -0.01986694f, 0.03777607f,  0.56661550f,  0.28896755f,  -0.04022174f,
    -0.01185068f, -0.03144731f, 0.03777607f,  0.00278718f,  -0.02921014f,
    -0.00213539f, -0.01185068f, -0.01986694f, -0.01610267f, -0.00624645f,
};
const float kernel11[25] = {
    -0.00213539f, -0.01185068f, -0.01986694f, -0.01610267f, -0.00624645f,
    -0.01185068f, -0.03144731f, 0.03777607f,  0.00278718f,  -0.02921014f,
    -0.01986694f, 0.03777607f,  0.56661550f,  0.28896755f,  -0.04022174f,
    -0.01610267f, 0.00278718f,  0.28896755f,  0.14111091f,  -0.03452303f,
    -0.00624645f, -0.02921014f, -0.04022174f, -0.03452303f, -0.01716200f,
};

// Does exactly the same as the Upsampler in dec_upsampler for 2x2 pixels, with
// default CustomTransformData.
// TODO(lode): use Upsampler instead. However, it requires pre-initialization
// and padding on the left side of the image which requires refactoring the
// other code using this.
void UpsampleImage(const ImageF& input, ImageF* output) {
  int64_t xsize = input.xsize();
  int64_t ysize = input.ysize();
  int64_t xsize2 = output->xsize();
  int64_t ysize2 = output->ysize();
  for (int64_t y = 0; y < ysize2; y++) {
    for (int64_t x = 0; x < xsize2; x++) {
      const auto* kernel = kernel00;
      if ((x & 1) && (y & 1)) {
        kernel = kernel11;
      } else if (x & 1) {
        kernel = kernel10;
      } else if (y & 1) {
        kernel = kernel01;
      }
      float sum = 0;
      int64_t x2 = x / 2;
      int64_t y2 = y / 2;

      // get min and max values of the original image in the support
      float min = std::numeric_limits<float>::max();
      float max = std::numeric_limits<float>::min();

      for (int64_t ky = 0; ky < kSize; ky++) {
        for (int64_t kx = 0; kx < kSize; kx++) {
          int64_t xi = x2 - kSize / 2 + kx;
          int64_t yi = y2 - kSize / 2 + ky;
          if (xi < 0) xi = 0;
          if (xi >= xsize) xi = input.xsize() - 1;
          if (yi < 0) yi = 0;
          if (yi >= ysize) yi = input.ysize() - 1;
          min = std::min<float>(min, input.Row(yi)[xi]);
          max = std::max<float>(max, input.Row(yi)[xi]);
        }
      }

      for (int64_t ky = 0; ky < kSize; ky++) {
        for (int64_t kx = 0; kx < kSize; kx++) {
          int64_t xi = x2 - kSize / 2 + kx;
          int64_t yi = y2 - kSize / 2 + ky;
          if (xi < 0) xi = 0;
          if (xi >= xsize) xi = input.xsize() - 1;
          if (yi < 0) yi = 0;
          if (yi >= ysize) yi = input.ysize() - 1;
          sum += input.Row(yi)[xi] * kernel[ky * kSize + kx];
        }
      }
      output->Row(y)[x] = sum;
      if (output->Row(y)[x] < min) output->Row(y)[x] = min;
      if (output->Row(y)[x] > max) output->Row(y)[x] = max;
    }
  }
}

// Returns the derivative of Upsampler, with respect to input pixel x2, y2, to
// output pixel x, y (ignoring the clamping).
float UpsamplerDeriv(int64_t x2, int64_t y2, int64_t x, int64_t y) {
  const auto* kernel = kernel00;
  if ((x & 1) && (y & 1)) {
    kernel = kernel11;
  } else if (x & 1) {
    kernel = kernel10;
  } else if (y & 1) {
    kernel = kernel01;
  }

  int64_t ix = x / 2;
  int64_t iy = y / 2;
  int64_t kx = x2 - ix + kSize / 2;
  int64_t ky = y2 - iy + kSize / 2;

  // This should not happen.
  if (kx < 0 || kx >= kSize || ky < 0 || ky >= kSize) return 0;

  return kernel[ky * kSize + kx];
}

// Apply the derivative of the Upsampler to the input, reversing the effect of
// its coefficients. The output image is 2x2 times smaller than the input.
void AntiUpsample(const ImageF& input, ImageF* d) {
  int64_t xsize = input.xsize();
  int64_t ysize = input.ysize();
  int64_t xsize2 = d->xsize();
  int64_t ysize2 = d->ysize();
  int64_t k0 = kSize - 1;
  int64_t k1 = kSize;
  for (int64_t y2 = 0; y2 < ysize2; ++y2) {
    auto* row = d->Row(y2);
    for (int64_t x2 = 0; x2 < xsize2; ++x2) {
      int64_t x0 = x2 * 2 - k0;
      if (x0 < 0) x0 = 0;
      int64_t x1 = x2 * 2 + k1 + 1;
      if (x1 > xsize) x1 = xsize;
      int64_t y0 = y2 * 2 - k0;
      if (y0 < 0) y0 = 0;
      int64_t y1 = y2 * 2 + k1 + 1;
      if (y1 > ysize) y1 = ysize;

      float sum = 0;
      for (int64_t y = y0; y < y1; ++y) {
        const auto* row_in = input.Row(y);
        for (int64_t x = x0; x < x1; ++x) {
          double deriv = UpsamplerDeriv(x2, y2, x, y);
          sum += deriv * row_in[x];
        }
      }
      row[x2] = sum;
    }
  }
}

// Element-wise multiplies two images.
template <typename T>
Status ElwiseMul(const Plane<T>& image1, const Plane<T>& image2,
                 Plane<T>* out) {
  const size_t xsize = image1.xsize();
  const size_t ysize = image1.ysize();
  JXL_ENSURE(xsize == image2.xsize());
  JXL_ENSURE(ysize == image2.ysize());
  JXL_ENSURE(xsize == out->xsize());
  JXL_ENSURE(ysize == out->ysize());
  for (size_t y = 0; y < ysize; ++y) {
    const T* const JXL_RESTRICT row1 = image1.Row(y);
    const T* const JXL_RESTRICT row2 = image2.Row(y);
    T* const JXL_RESTRICT row_out = out->Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      row_out[x] = row1[x] * row2[x];
    }
  }
  return true;
}

// Element-wise divides two images.
template <typename T>
Status ElwiseDiv(const Plane<T>& image1, const Plane<T>& image2,
                 Plane<T>* out) {
  const size_t xsize = image1.xsize();
  const size_t ysize = image1.ysize();
  JXL_ENSURE(xsize == image2.xsize());
  JXL_ENSURE(ysize == image2.ysize());
  JXL_ENSURE(xsize == out->xsize());
  JXL_ENSURE(ysize == out->ysize());
  for (size_t y = 0; y < ysize; ++y) {
    const T* const JXL_RESTRICT row1 = image1.Row(y);
    const T* const JXL_RESTRICT row2 = image2.Row(y);
    T* const JXL_RESTRICT row_out = out->Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      row_out[x] = row1[x] / row2[x];
    }
  }
  return true;
}

void ReduceRinging(const ImageF& initial, const ImageF& mask, ImageF& down) {
  int64_t xsize2 = down.xsize();
  int64_t ysize2 = down.ysize();

  for (size_t y = 0; y < down.ysize(); y++) {
    const float* row_mask = mask.Row(y);
    float* row_out = down.Row(y);
    for (size_t x = 0; x < down.xsize(); x++) {
      float v = down.Row(y)[x];
      float min = initial.Row(y)[x];
      float max = initial.Row(y)[x];
      for (int64_t yi = -1; yi < 2; yi++) {
        for (int64_t xi = -1; xi < 2; xi++) {
          int64_t x2 = static_cast<int64_t>(x) + xi;
          int64_t y2 = static_cast<int64_t>(y) + yi;
          if (x2 < 0 || y2 < 0 || x2 >= xsize2 || y2 >= ysize2) continue;
          min = std::min<float>(min, initial.Row(y2)[x2]);
          max = std::max<float>(max, initial.Row(y2)[x2]);
        }
      }

      row_out[x] = v;

      // Clamp the pixel within the value  of a small area to prevent ringning.
      // The mask determines how much to clamp, clamp more to reduce more
      // ringing in smooth areas, clamp less in noisy areas to get more
      // sharpness. Higher mask_multiplier gives less clamping, so less
      // ringing reduction.
      const constexpr float mask_multiplier = 2;
      float a = row_mask[x] * mask_multiplier;
      float clip_min = min - a;
      float clip_max = max + a;
      if (row_out[x] < clip_min) row_out[x] = clip_min;
      if (row_out[x] > clip_max) row_out[x] = clip_max;
    }
  }
}

// TODO(lode): move this to a separate file enc_downsample.cc
Status DownsampleImage2_Iterative(const ImageF& orig, ImageF* output) {
  int64_t xsize = orig.xsize();
  int64_t ysize = orig.ysize();
  int64_t xsize2 = DivCeil(orig.xsize(), 2);
  int64_t ysize2 = DivCeil(orig.ysize(), 2);
  JxlMemoryManager* memory_manager = orig.memory_manager();

  JXL_ASSIGN_OR_RETURN(ImageF box_downsample,
                       ImageF::Create(memory_manager, xsize, ysize));
  JXL_RETURN_IF_ERROR(CopyImageTo(orig, &box_downsample));
  JXL_ASSIGN_OR_RETURN(box_downsample, DownsampleImage(box_downsample, 2));
  JXL_ASSIGN_OR_RETURN(ImageF mask,
                       ImageF::Create(memory_manager, box_downsample.xsize(),
                                      box_downsample.ysize()));
  CreateMask(box_downsample, mask);

  JXL_RETURN_IF_ERROR(output->ShrinkTo(xsize2, ysize2));

  // Initial result image using the sharper downsampling.
  // Allocate extra space to avoid a reallocation when padding.
  JXL_ASSIGN_OR_RETURN(
      ImageF initial,
      ImageF::Create(memory_manager, DivCeil(orig.xsize(), 2) + kBlockDim,
                     DivCeil(orig.ysize(), 2) + kBlockDim));
  JXL_RETURN_IF_ERROR(initial.ShrinkTo(initial.xsize() - kBlockDim,
                                       initial.ysize() - kBlockDim));
  JXL_RETURN_IF_ERROR(DownsampleImage2_Sharper(orig, &initial));

  JXL_ASSIGN_OR_RETURN(
      ImageF down,
      ImageF::Create(memory_manager, initial.xsize(), initial.ysize()));
  JXL_RETURN_IF_ERROR(CopyImageTo(initial, &down));
  JXL_ASSIGN_OR_RETURN(ImageF up, ImageF::Create(memory_manager, xsize, ysize));
  JXL_ASSIGN_OR_RETURN(ImageF corr,
                       ImageF::Create(memory_manager, xsize, ysize));
  JXL_ASSIGN_OR_RETURN(ImageF corr2,
                       ImageF::Create(memory_manager, xsize2, ysize2));

  // In the weights map, relatively higher values will allow less ringing but
  // also less sharpness. With all constant values, it optimizes equally
  // everywhere. Even in this case, the weights2 computed from
  // this is still used and differs at the borders of the image.
  // TODO(lode): Make use of the weights field for anti-ringing and clamping,
  // the values are all set to 1 for now, but it is intended to be used for
  // reducing ringing based on the mask, and taking clamping into account.
  JXL_ASSIGN_OR_RETURN(ImageF weights,
                       ImageF::Create(memory_manager, xsize, ysize));
  for (size_t y = 0; y < weights.ysize(); y++) {
    auto* row = weights.Row(y);
    for (size_t x = 0; x < weights.xsize(); x++) {
      row[x] = 1;
    }
  }
  JXL_ASSIGN_OR_RETURN(ImageF weights2,
                       ImageF::Create(memory_manager, xsize2, ysize2));
  AntiUpsample(weights, &weights2);

  const size_t num_it = 3;
  for (size_t it = 0; it < num_it; ++it) {
    UpsampleImage(down, &up);
    JXL_ASSIGN_OR_RETURN(corr, LinComb<float>(1, orig, -1, up));
    JXL_RETURN_IF_ERROR(ElwiseMul(corr, weights, &corr));
    AntiUpsample(corr, &corr2);
    JXL_RETURN_IF_ERROR(ElwiseDiv(corr2, weights2, &corr2));

    JXL_ASSIGN_OR_RETURN(down, LinComb<float>(1, down, 1, corr2));
  }

  ReduceRinging(initial, mask, down);

  // can't just use CopyImage, because the output image was prepared with
  // padding.
  for (size_t y = 0; y < down.ysize(); y++) {
    for (size_t x = 0; x < down.xsize(); x++) {
      float v = down.Row(y)[x];
      output->Row(y)[x] = v;
    }
  }
  return true;
}

}  // namespace

Status DownsampleImage2_Iterative(Image3F* opsin) {
  JxlMemoryManager* memory_manager = opsin->memory_manager();
  // Allocate extra space to avoid a reallocation when padding.
  JXL_ASSIGN_OR_RETURN(
      Image3F downsampled,
      Image3F::Create(memory_manager, DivCeil(opsin->xsize(), 2) + kBlockDim,
                      DivCeil(opsin->ysize(), 2) + kBlockDim));
  JXL_RETURN_IF_ERROR(downsampled.ShrinkTo(downsampled.xsize() - kBlockDim,
                                           downsampled.ysize() - kBlockDim));

  JXL_ASSIGN_OR_RETURN(
      Image3F rgb,
      Image3F::Create(memory_manager, opsin->xsize(), opsin->ysize()));
  OpsinParams opsin_params;  // TODO(user): use the ones that are actually used
  opsin_params.Init(kDefaultIntensityTarget);
  JXL_RETURN_IF_ERROR(
      OpsinToLinear(*opsin, Rect(rgb), nullptr, &rgb, opsin_params));

  JXL_ASSIGN_OR_RETURN(
      ImageF mask,
      ImageF::Create(memory_manager, opsin->xsize(), opsin->ysize()));
  ButteraugliParams butter_params;
  JXL_ASSIGN_OR_RETURN(std::unique_ptr<ButteraugliComparator> butter,
                       ButteraugliComparator::Make(rgb, butter_params));
  JXL_RETURN_IF_ERROR(butter->Mask(&mask));
  JXL_ASSIGN_OR_RETURN(
      ImageF mask_fuzzy,
      ImageF::Create(memory_manager, opsin->xsize(), opsin->ysize()));

  for (size_t c = 0; c < 3; c++) {
    JXL_RETURN_IF_ERROR(
        DownsampleImage2_Iterative(opsin->Plane(c), &downsampled.Plane(c)));
  }
  *opsin = std::move(downsampled);
  return true;
}

StatusOr<Image3F> ReconstructImage(
    const FrameHeader& orig_frame_header, const PassesSharedState& shared,
    const std::vector<std::unique_ptr<ACImage>>& coeffs, ThreadPool* pool) {
  const FrameDimensions& frame_dim = shared.frame_dim;
  JxlMemoryManager* memory_manager = shared.memory_manager;

  FrameHeader frame_header = orig_frame_header;
  frame_header.UpdateFlag(shared.image_features.patches.HasAny(),
                          FrameHeader::kPatches);
  frame_header.UpdateFlag(shared.image_features.splines.HasAny(),
                          FrameHeader::kSplines);
  frame_header.color_transform = ColorTransform::kNone;

  auto metadata = jxl::make_unique<CodecMetadata>();
  *metadata = *frame_header.nonserialized_metadata;
  metadata->m.extra_channel_info.clear();
  metadata->m.num_extra_channels = metadata->m.extra_channel_info.size();
  frame_header.nonserialized_metadata = metadata.get();
  frame_header.extra_channel_upsampling.clear();

  const bool is_gray = shared.metadata->m.color_encoding.IsGray();
  auto dec_state = jxl::make_unique<PassesDecoderState>(memory_manager);
  JXL_RETURN_IF_ERROR(
      dec_state->output_encoding_info.SetFromMetadata(*shared.metadata));
  JXL_RETURN_IF_ERROR(dec_state->output_encoding_info.MaybeSetColorEncoding(
      ColorEncoding::LinearSRGB(is_gray)));
  dec_state->shared = &shared;
  JXL_RETURN_IF_ERROR(dec_state->Init(frame_header));

  ImageBundle decoded(memory_manager, &shared.metadata->m);
  decoded.origin = frame_header.frame_origin;
  JXL_ASSIGN_OR_RETURN(
      Image3F tmp,
      Image3F::Create(memory_manager, frame_dim.xsize, frame_dim.ysize));
  JXL_RETURN_IF_ERROR(decoded.SetFromImage(
      std::move(tmp), dec_state->output_encoding_info.color_encoding));

  PassesDecoderState::PipelineOptions options;
  options.use_slow_render_pipeline = false;
  options.coalescing = false;
  options.render_spotcolors = false;
  options.render_noise = false;

  JXL_RETURN_IF_ERROR(dec_state->PreparePipeline(
      frame_header, &shared.metadata->m, &decoded, options));

  AlignedArray<GroupDecCache> group_dec_caches;
  const auto allocate_storage = [&](const size_t num_threads) -> Status {
    JXL_RETURN_IF_ERROR(
        dec_state->render_pipeline->PrepareForThreads(num_threads,
                                                      /*use_group_ids=*/false));
    JXL_ASSIGN_OR_RETURN(group_dec_caches, AlignedArray<GroupDecCache>::Create(
                                               memory_manager, num_threads));
    return true;
  };
  const auto process_group = [&](const uint32_t group_index,
                                 const size_t thread) -> Status {
    if (frame_header.loop_filter.epf_iters > 0) {
      JXL_RETURN_IF_ERROR(ComputeSigma(frame_header.loop_filter,
                                       frame_dim.BlockGroupRect(group_index),
                                       dec_state.get()));
    }
    RenderPipelineInput input =
        dec_state->render_pipeline->GetInputBuffers(group_index, thread);
    JXL_RETURN_IF_ERROR(DecodeGroupForRoundtrip(
        frame_header, coeffs, group_index, dec_state.get(),
        &group_dec_caches[thread], thread, input, nullptr, nullptr));
    JXL_RETURN_IF_ERROR(input.Done());
    return true;
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, frame_dim.num_groups, allocate_storage,
                                process_group, "ReconstructImage"));
  return std::move(*decoded.color());
}

float ComputeBlockL2Distance(const Image3F& a, const Image3F& b,
                             const ImageF& mask1x1, size_t by, size_t bx,
                             float* JXL_RESTRICT out_bdry_err = nullptr,
                             float* JXL_RESTRICT out_grad_a = nullptr,
                             float* JXL_RESTRICT out_grad_b = nullptr) {
  Rect rect(bx * kBlockDim, by * kBlockDim, kBlockDim, kBlockDim, a.xsize(),
            a.ysize());
  const size_t xsize = rect.xsize();
  const size_t ysize = rect.ysize();
  const size_t x0 = rect.x0();
  const size_t y0 = rect.y0();
  const bool has_left = (out_bdry_err != nullptr) && (x0 > 0);
  const bool has_right = (out_bdry_err != nullptr) && (x0 + xsize < a.xsize());
  const bool has_top = (out_bdry_err != nullptr) && (y0 > 0);
  const bool has_bottom = (out_bdry_err != nullptr) && (y0 + ysize < a.ysize());
  float err2_masked[3] = {0.0f};
  float err2_unmasked[3] = {0.0f};
  float bdry_err2[3] = {0.0f};
  float grad_a2[3] = {0.0f};
  float grad_b2[3] = {0.0f};
  float sum_mask2 = 0.0f;
  const bool full_8x8 = (xsize == 8 && ysize == 8);
  float cell_a[3][4] = {};
  float cell_b[3][4] = {};
  float cell_a2[3][4] = {};
  float cell_b2[3][4] = {};
  float cell_e2[3][4] = {};
  static constexpr float kSsimScale[3] = {14.0f, 1.0f, 1.0f};
  for (size_t y = 0; y < ysize; ++y) {
    const float* row_a[3] = {
        rect.ConstPlaneRow(a, 0, y),
        rect.ConstPlaneRow(a, 1, y),
        rect.ConstPlaneRow(a, 2, y),
    };
    const float* row_b[3] = {
        rect.ConstPlaneRow(b, 0, y),
        rect.ConstPlaneRow(b, 1, y),
        rect.ConstPlaneRow(b, 2, y),
    };
    const float* row_a_next[3] = {
        (y + 1 < ysize || has_bottom) ? a.ConstPlaneRow(0, y0 + y + 1) + x0
                                      : nullptr,
        (y + 1 < ysize || has_bottom) ? a.ConstPlaneRow(1, y0 + y + 1) + x0
                                      : nullptr,
        (y + 1 < ysize || has_bottom) ? a.ConstPlaneRow(2, y0 + y + 1) + x0
                                      : nullptr,
    };
    const float* row_b_next[3] = {
        (y + 1 < ysize || has_bottom) ? b.ConstPlaneRow(0, y0 + y + 1) + x0
                                      : nullptr,
        (y + 1 < ysize || has_bottom) ? b.ConstPlaneRow(1, y0 + y + 1) + x0
                                      : nullptr,
        (y + 1 < ysize || has_bottom) ? b.ConstPlaneRow(2, y0 + y + 1) + x0
                                      : nullptr,
    };
    const float* row_a_prev[3] = {
        (y == 0 && has_top) ? a.ConstPlaneRow(0, y0 - 1) + x0 : nullptr,
        (y == 0 && has_top) ? a.ConstPlaneRow(1, y0 - 1) + x0 : nullptr,
        (y == 0 && has_top) ? a.ConstPlaneRow(2, y0 - 1) + x0 : nullptr,
    };
    const float* row_b_prev[3] = {
        (y == 0 && has_top) ? b.ConstPlaneRow(0, y0 - 1) + x0 : nullptr,
        (y == 0 && has_top) ? b.ConstPlaneRow(1, y0 - 1) + x0 : nullptr,
        (y == 0 && has_top) ? b.ConstPlaneRow(2, y0 - 1) + x0 : nullptr,
    };
    const float* row_mask = rect.ConstRow(mask1x1, y);
    for (size_t x = 0; x < xsize; ++x) {
      float mask = row_mask[x];
      float mask2 = mask * mask;
      sum_mask2 += mask2;
      if (full_8x8) {
        const size_t q = ((y >> 2) << 1) + (x >> 2);
        const float oa0 = kSsimScale[0] * row_a[0][x];
        const float ob0 = kSsimScale[0] * row_b[0][x];
        const float oa1 = row_a[1][x];
        const float ob1 = row_b[1][x];
        const float oa2 = row_a[2][x] - oa1;
        const float ob2 = row_b[2][x] - ob1;
        const float oa[3] = {oa0, oa1, oa2};
        const float ob[3] = {ob0, ob1, ob2};
        for (int c = 0; c < 3; ++c) {
          const float oe = oa[c] - ob[c];
          cell_a[c][q] += oa[c];
          cell_b[c][q] += ob[c];
          cell_a2[c][q] += oa[c] * oa[c];
          cell_b2[c][q] += ob[c] * ob[c];
          cell_e2[c][q] += oe * oe;
        }
      }
      for (int i = 0; i < 3; ++i) {
        float diff = row_a[i][x] - row_b[i][x];
        float diff2 = diff * diff;
        err2_masked[i] += mask2 * diff2;
        err2_unmasked[i] += diff2;
        if (x + 1 < xsize) {
          float ga = row_a[i][x + 1] - row_a[i][x];
          float gb = row_b[i][x + 1] - row_b[i][x];
          grad_a2[i] += ga * ga;
          grad_b2[i] += gb * gb;
        }
        if (y + 1 < ysize) {
          float ga = row_a_next[i][x] - row_a[i][x];
          float gb = row_b_next[i][x] - row_b[i][x];
          grad_a2[i] += ga * ga;
          grad_b2[i] += gb * gb;
        }
        if (x == 0 && has_left) {
          float diff_l = row_a[i][-1] - row_b[i][-1];
          float db = diff - diff_l;
          bdry_err2[i] += mask2 * db * db;
        }
        if (x + 1 == xsize && has_right) {
          float diff_r = row_a[i][xsize] - row_b[i][xsize];
          float db = diff - diff_r;
          bdry_err2[i] += mask2 * db * db;
        }
        if (y == 0 && has_top) {
          float diff_t = row_a_prev[i][x] - row_b_prev[i][x];
          float db = diff - diff_t;
          bdry_err2[i] += mask2 * db * db;
        }
        if (y + 1 == ysize && has_bottom) {
          float diff_d = row_a_next[i][x] - row_b_next[i][x];
          float db = diff - diff_d;
          bdry_err2[i] += mask2 * db * db;
        }
      }
    }
  }
  const float mean_mask2 =
      sum_mask2 / static_cast<float>(xsize * ysize);
  constexpr float kUnmaskedMix = 0.22f;
  float err2[3];
  for (int i = 0; i < 3; ++i) {
    err2[i] = (1.0f - kUnmaskedMix) * err2_masked[i] +
              kUnmaskedMix * mean_mask2 * err2_unmasked[i];
  }
  static const double kW[] = {
      12.339445295782363,
      1.0,
      0.2,
  };
  if (out_bdry_err != nullptr) {
    *out_bdry_err =
        kW[0] * bdry_err2[0] + kW[1] * bdry_err2[1] + kW[2] * bdry_err2[2];
  }
  if (out_grad_a != nullptr && out_grad_b != nullptr) {
    *out_grad_a = mean_mask2 * (kW[0] * grad_a2[0] + kW[1] * grad_a2[1] +
                                kW[2] * grad_a2[2]);
    *out_grad_b = mean_mask2 * (kW[0] * grad_b2[0] + kW[1] * grad_b2[1] +
                                kW[2] * grad_b2[2]);
  }
  float retval = kW[0] * err2[0] + kW[1] * err2[1] + kW[2] * err2[2];
  if (full_8x8) {
    static constexpr float kW_ssim[3] = {0.06f, 1.00f, 0.45f};
    float ssim_err = 0.0f;
    for (int c = 0; c < 3; ++c) {
      float mu_a[4];
      float mu_b[4];
      float c_err = 0.0f;
      for (size_t q = 0; q < 4; ++q) {
        mu_a[q] = 0.0625f * cell_a[c][q];
        mu_b[q] = 0.0625f * cell_b[c][q];
        const float mu_e = mu_a[q] - mu_b[q];
        const float sig_a2 =
            std::max(0.0f, 0.0625f * cell_a2[c][q] - mu_a[q] * mu_a[q]);
        const float sig_b2 =
            std::max(0.0f, 0.0625f * cell_b2[c][q] - mu_b[q] * mu_b[q]);
        const float sig_e2 =
            std::max(0.0f, 0.0625f * cell_e2[c][q] - mu_e * mu_e);
        const float denom = sig_a2 + sig_b2 + 0.0009f;
        const float two_cov_c2 = std::max(0.0f, denom - sig_e2);
        const float d_ssim = (sig_e2 + mu_e * mu_e * two_cov_c2) / denom;
        const float sd_a = std::sqrt(sig_a2);
        const float sd_b = std::sqrt(sig_b2);
        const float edge_d = (sd_b - sd_a) / (1.0f + sd_a);
        const float edge_pen =
            (edge_d < 0.0f) ? (1.20f * edge_d * edge_d) : (0.45f * edge_d * edge_d);
        c_err += 16.0f * (900.0f * d_ssim + 750.0f * edge_pen);
      }
      const float mu_a8 = 0.25f * (mu_a[0] + mu_a[1] + mu_a[2] + mu_a[3]);
      const float mu_b8 = 0.25f * (mu_b[0] + mu_b[1] + mu_b[2] + mu_b[3]);
      const float mu_e8 = mu_a8 - mu_b8;
      float s_a2 = 0.0f;
      float s_b2 = 0.0f;
      float s_e2 = 0.0f;
      for (size_t q = 0; q < 4; ++q) {
        const float da = mu_a[q] - mu_a8;
        const float db = mu_b[q] - mu_b8;
        const float de = da - db;
        s_a2 += da * da;
        s_b2 += db * db;
        s_e2 += de * de;
      }
      const float denom8 = 0.25f * (s_a2 + s_b2) + 0.0009f;
      const float two_cov8 = std::max(0.0f, denom8 - 0.25f * s_e2);
      const float d_ssim8 = (0.25f * s_e2 + mu_e8 * mu_e8 * two_cov8) / denom8;
      c_err = 0.45f * c_err + 0.55f * (64.0f * 900.0f * d_ssim8);
      ssim_err += kW_ssim[c] * c_err;
    }
    retval = 0.45f * retval + 0.55f * ssim_err;
  }
  return retval;
}

Status ComputeARHeuristics(const FrameHeader& frame_header,
                           PassesEncoderState* enc_state,
                           const Image3F& orig_opsin, const Rect& rect,
                           ThreadPool* pool) {
  const CompressParams& cparams = enc_state->cparams;
  PassesSharedState& shared = enc_state->shared;
  const FrameDimensions& frame_dim = shared.frame_dim;
  const ImageF& initial_quant_masking1x1 = enc_state->initial_quant_masking1x1;
  ImageB& epf_sharpness = shared.epf_sharpness;
  JxlMemoryManager* memory_manager = enc_state->memory_manager();

  float clamped_butteraugli = std::min(5.0f, cparams.butteraugli_distance);
  const bool ar_enable_rdo =
      cparams.speed_tier <= SpeedTier::kSquirrel &&
      !cparams.disable_perceptual_optimizations &&
      cparams.butteraugli_distance >= 0.35f &&
      cparams.butteraugli_distance <= 15.0f && cparams.resampling <= 1 &&
      cparams.ec_resampling <= 1 && cparams.gaborish != Override::kOff &&
      enc_state->progressive_splitter.GetNumPasses() == 1;
  if (frame_header.loop_filter.epf_iters == 0 && ar_enable_rdo) {
    FillPlane(static_cast<uint8_t>(0), &epf_sharpness, Rect(epf_sharpness));
    return true;
  }
  if (cparams.butteraugli_distance < kMinButteraugliForDynamicAR ||
      cparams.speed_tier > SpeedTier::kWombat ||
      !cparams.use_full_image_heuristics ||
      frame_header.loop_filter.epf_iters == 0 ||
      frame_header.upsampling != 1) {
    FillPlane(static_cast<uint8_t>(4), &epf_sharpness, Rect(epf_sharpness));
    return true;
  }

  std::vector<uint8_t> epf_steps;
  if (cparams.butteraugli_distance > 4.5f) {
    epf_steps.push_back(0);
    epf_steps.push_back(4);
  } else {
    epf_steps.push_back(0);
    epf_steps.push_back(2);
    epf_steps.push_back(7);
  }
  static const int kNumEPFVals = 8;
  size_t epf_steps_lut[kNumEPFVals] = {0};
  {
    for (size_t i = 0; i < epf_steps.size(); ++i) {
      epf_steps_lut[epf_steps[i]] = i;
    }
  }
  std::array<ImageF, kNumEPFVals> error_images;
  ImageF bdry_err0;
  JXL_ASSIGN_OR_RETURN(bdry_err0,
                       ImageF::Create(memory_manager, frame_dim.xsize_blocks,
                                      frame_dim.ysize_blocks));
  ImageF target_grad_int;
  JXL_ASSIGN_OR_RETURN(target_grad_int,
                       ImageF::Create(memory_manager, frame_dim.xsize_blocks,
                                      frame_dim.ysize_blocks));
  constexpr float kBdryWeight = 0.08f;
  constexpr float kTexWeight7 = 0.012f;
  const float tex_weight_2 =
      0.008f * Clamp1((1.15f - clamped_butteraugli) / 0.65f, 0.0f, 1.0f);
  for (uint8_t val : epf_steps) {
    FillPlane(val, &epf_sharpness, Rect(epf_sharpness));
    JXL_ASSIGN_OR_RETURN(
        Image3F decoded,
        ReconstructImage(frame_header, shared, enc_state->coeffs, pool));
    JXL_ASSIGN_OR_RETURN(error_images[val],
                         ImageF::Create(memory_manager, frame_dim.xsize_blocks,
                                        frame_dim.ysize_blocks));
    for (size_t by = 0; by < frame_dim.ysize_blocks; by++) {
      float* error_row = error_images[val].Row(by);
      float* bdry0_row = bdry_err0.Row(by);
      float* target_grad_row = target_grad_int.Row(by);
      for (size_t bx = 0; bx < frame_dim.xsize_blocks; bx++) {
        float bdry_b = 0.0f;
        float grad_a = 0.0f;
        float grad_b = 0.0f;
        float err = ComputeBlockL2Distance(
            orig_opsin, decoded, initial_quant_masking1x1, by, bx, &bdry_b,
            &grad_a, &grad_b);
        if (val == 0) {
          bdry0_row[bx] = bdry_b;
          target_grad_row[bx] = std::min(grad_a, grad_b);
        } else {
          const float tw = (val >= 7) ? kTexWeight7 : tex_weight_2;
          err += kBdryWeight * (bdry_b - bdry0_row[bx]) +
                 tw * std::max(0.0f, target_grad_row[bx] - grad_b);
        }
        error_row[bx] = err;
      }
    }
  }
  double sum_sqrt_d0 = 0.0;
  for (size_t by = 0; by < frame_dim.ysize_blocks; by++) {
    const float* row0 = error_images[0].ConstRow(by);
    for (size_t bx = 0; bx < frame_dim.xsize_blocks; bx++) {
      sum_sqrt_d0 += std::sqrt(row0[bx]);
    }
  }
  const float mean_sqrt_d0 =
      sum_sqrt_d0 /
      std::max<size_t>(1, frame_dim.xsize_blocks * frame_dim.ysize_blocks);
  const float ref_d0 = mean_sqrt_d0 * mean_sqrt_d0;
  constexpr float kLocalRef = 0.84f;
  const float global_ref =
      0.15f / std::pow(clamped_butteraugli, 0.35f);
  std::vector<std::vector<size_t>> histo(9, std::vector<size_t>(kNumEPFVals));
  std::vector<size_t> totals(9, 1);
  std::vector<std::vector<size_t>> bctx_histo(4,
                                              std::vector<size_t>(kNumEPFVals));
  std::vector<size_t> bctx_totals(4, 1);
  static const float kFavorNoSmoothing = 0.99;
  for (size_t by = 0; by < frame_dim.ysize_blocks; by++) {
    uint8_t* JXL_RESTRICT out_row = epf_sharpness.Row(by);
    uint8_t* JXL_RESTRICT prev_row = epf_sharpness.Row(by > 0 ? by - 1 : 0);
    const float* row0 = error_images[0].ConstRow(by);
    for (size_t bx = 0; bx < frame_dim.xsize_blocks; bx++) {
      const float d_ref = kLocalRef * row0[bx] + global_ref * ref_d0;
      uint8_t best_val = 0;
      float best_error = std::numeric_limits<float>::max();
      uint8_t top_val = by > 0 ? prev_row[bx] : 0;
      uint8_t left_val = bx > 0 ? out_row[bx - 1] : 0;
      float top_mul = 0.985f * (top_val == 0 ? kFavorNoSmoothing : 1.0f);
      float left_mul = 0.985f * (left_val == 0 ? kFavorNoSmoothing : 1.0f);
      float top_error =
          error_images[top_val].Row(by)[bx] + d_ref * (top_mul - 1.0f);
      float left_error =
          error_images[left_val].Row(by)[bx] + d_ref * (left_mul - 1.0f);
      for (uint8_t val : epf_steps) {
        float m = (val == 0) ? kFavorNoSmoothing : 1.0f;
        float error = error_images[val].Row(by)[bx] + d_ref * (m - 1.0f);
        if (error < best_error) {
          best_val = val;
          best_error = error;
        }
      }
      if (best_error < std::min(top_error, left_error)) {
        out_row[bx] = best_val;
      } else if (top_error < left_error) {
        out_row[bx] = top_val;
      } else {
        out_row[bx] = left_val;
      }
      int context = epf_steps_lut[top_val] * 3 + epf_steps_lut[left_val];
      int bctx = (top_val > 0 ? 2 : 0) + (left_val > 0 ? 1 : 0);
      ++histo[context][out_row[bx]];
      ++totals[context];
      ++bctx_histo[bctx][out_row[bx]];
      ++bctx_totals[bctx];
    }
  }
  const float c3base = 0.98017198824148288;
  const float c3clamp = 0.85970338919928291;
  const float c3 = std::max(c3clamp, std::pow(c3base, clamped_butteraugli));
  static const float c5 = 0.1087690359555803;
  float mul[3 * 3 * 3] = {0};
  for (uint8_t top_val : epf_steps) {
    for (uint8_t left_val : epf_steps) {
      for (uint8_t val : epf_steps) {
        int context = epf_steps_lut[top_val] * 3 + epf_steps_lut[left_val];
        int bctx = (top_val > 0 ? 2 : 0) + (left_val > 0 ? 1 : 0);
        const auto& ctx_histo = histo[context];
        const int mulix = epf_steps_lut[val] + 3 * context;
        const float ratio_9 = static_cast<float>(ctx_histo[val]) /
                              static_cast<float>(totals[context]);
        const float ratio_4 = static_cast<float>(bctx_histo[bctx][val]) /
                              static_cast<float>(bctx_totals[bctx]);
        const float ratio = 0.65f * ratio_9 + 0.35f * ratio_4;
        mul[mulix] =
            1.0f / (1.0f + c5 * std::log1p(ratio) / clamped_butteraugli);
        if (val == top_val && val == left_val) {
          mul[mulix] *= 0.988f;
        } else if (val == top_val || val == left_val) {
          mul[mulix] *= 0.994f;
        }
        if (val == 0) {
          mul[mulix] *= c3;
        }
      }
    }
  }
  double total_mod_err = 0.0;
  double sum_d_ref = 0.0;
  size_t final_bctx_histo[4][kNumEPFVals] = {};
  size_t final_bctx_totals[4] = {};
  size_t val_counts[kNumEPFVals] = {};
  for (size_t by = 0; by < frame_dim.ysize_blocks; by++) {
    uint8_t* JXL_RESTRICT out_row = epf_sharpness.Row(by);
    uint8_t* JXL_RESTRICT prev_row = epf_sharpness.Row(by > 0 ? by - 1 : 0);
    const float* row0 = error_images[0].ConstRow(by);
    for (size_t bx = 0; bx < frame_dim.xsize_blocks; bx++) {
      const float d_ref = kLocalRef * row0[bx] + global_ref * ref_d0;
      uint8_t best_val = 0;
      float best_error = std::numeric_limits<float>::max();
      uint8_t top_val = by > 0 ? prev_row[bx] : 0;
      uint8_t left_val = bx > 0 ? out_row[bx - 1] : 0;
      int context = epf_steps_lut[top_val] * 3 + epf_steps_lut[left_val];
      for (uint8_t val : epf_steps) {
        int mulix = epf_steps_lut[val] + 3 * context;
        float error =
            error_images[val].Row(by)[bx] + d_ref * (mul[mulix] - 1.0f);
        if (error < best_error) {
          best_val = val;
          best_error = error;
        }
      }
      out_row[bx] = best_val;
      total_mod_err += error_images[best_val].Row(by)[bx];
      sum_d_ref += d_ref;
      const int bctx = (top_val > 0 ? 2 : 0) + (left_val > 0 ? 1 : 0);
      final_bctx_histo[bctx][best_val]++;
      final_bctx_totals[bctx]++;
      val_counts[best_val]++;
    }
  }

  size_t num_unique_vals = 0;
  for (uint8_t val : epf_steps) {
    if (val_counts[val] > 0) num_unique_vals++;
  }
  if (num_unique_vals > 1 &&
      frame_dim.xsize_blocks * frame_dim.ysize_blocks > 0) {
    double epf_bits = 14.0;
    for (int bctx = 0; bctx < 4; ++bctx) {
      if (final_bctx_totals[bctx] == 0) continue;
      const double inv_tot =
          1.0 / static_cast<double>(final_bctx_totals[bctx]);
      for (uint8_t val : epf_steps) {
        const size_t cnt = final_bctx_histo[bctx][val];
        if (cnt > 0) {
          epf_bits -=
              static_cast<double>(cnt) * std::log2(static_cast<double>(cnt) * inv_tot);
        }
      }
    }
    const double mean_d_ref =
        sum_d_ref /
        static_cast<double>(frame_dim.xsize_blocks * frame_dim.ysize_blocks);
    const double lambda_bit =
        (0.018 / static_cast<double>(clamped_butteraugli)) * mean_d_ref;
    uint8_t best_const_val = 0;
    double best_const_err = std::numeric_limits<double>::max();
    for (uint8_t val : epf_steps) {
      double c_err = 0.0;
      for (size_t by = 0; by < frame_dim.ysize_blocks; ++by) {
        const float* erow = error_images[val].ConstRow(by);
        for (size_t bx = 0; bx < frame_dim.xsize_blocks; ++bx) {
          c_err += erow[bx];
        }
      }
      if (c_err < best_const_err) {
        best_const_err = c_err;
        best_const_val = val;
      }
    }
    if (best_const_err <= total_mod_err + lambda_bit * epf_bits) {
      FillPlane(best_const_val, &epf_sharpness, Rect(epf_sharpness));
    }
  }

  return true;
}

Status BuildPass1TokenCostModel(const FrameHeader& frame_header,
                                PassesEncoderState* enc_state,
                                const Image3F& opsin, const Rect& rect,
                                float cost_delta, ThreadPool* pool,
                                AcTokenCostModel* token_cost_model) {
  const CompressParams& cparams = enc_state->cparams;
  PassesSharedState& shared = enc_state->shared;
  const FrameDimensions& frame_dim = shared.frame_dim;
  ImageI& raw_quant_field = shared.raw_quant_field;
  AcStrategyImage& ac_strategy = shared.ac_strategy;
  BlockCtxMap& block_ctx_map = shared.block_ctx_map;
  JxlMemoryManager* memory_manager = enc_state->memory_manager();

  FindBestBlockEntropyModel(cparams, raw_quant_field, ac_strategy,
                            &block_ctx_map);
  if (enc_state->coeffs.empty()) {
    JXL_ASSIGN_OR_RETURN(
        std::unique_ptr<ACImageT<int32_t>> coeffs,
        ACImageT<int32_t>::Make(memory_manager, kGroupDim * kGroupDim,
                                frame_dim.num_groups));
    enc_state->coeffs.emplace_back(std::move(coeffs));
  }
  JXL_ASSIGN_OR_RETURN(
      Image3F pass1_dc,
      Image3F::Create(memory_manager, frame_dim.xsize_blocks,
                      frame_dim.ysize_blocks));
  const auto compute_pass1_coeffs = [&](size_t group_idx,
                                        size_t /*thread*/) -> Status {
    return ComputeCoefficients(group_idx, enc_state, opsin, rect, &pass1_dc);
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, frame_dim.num_groups,
                                ThreadPool::NoInit, compute_pass1_coeffs,
                                "Pass1 ComputeCoeffs"));

  token_cost_model->coeff_orders.resize(kCoeffOrderMaxSize);
  auto used_orders_info =
      ComputeUsedOrders(cparams.speed_tier, ac_strategy, Rect(raw_quant_field));
  uint32_t pass1_used_orders = 0;
  constexpr uint32_t kAllOrdersMask = (1u << kNumOrders) - 1u;
  JXL_RETURN_IF_ERROR(ComputeCoeffOrder(
      cparams.speed_tier, *enc_state->coeffs[0], ac_strategy, frame_dim,
      pass1_used_orders, /*prev_used_acs=*/0, kAllOrdersMask,
      used_orders_info.second, token_cost_model->coeff_orders.data()));

  ZeroFillImage(&shared.quant_dc);
  JXL_ASSIGN_OR_RETURN(
      token_cost_model->pass1_nzeros,
      Image3I::Create(memory_manager, frame_dim.xsize_blocks,
                      frame_dim.ysize_blocks));
  std::vector<std::vector<Token>> pass1_tokens(frame_dim.num_groups);
  std::vector<Image3I> pass1_tmp_nzeros;
  const auto tok_init = [&](const size_t num_threads) -> Status {
    pass1_tmp_nzeros.resize(num_threads);
    return true;
  };
  const auto tok_group = [&](const uint32_t group_index,
                             const size_t thread) -> Status {
    if (pass1_tmp_nzeros[thread].xsize() == 0) {
      JXL_ASSIGN_OR_RETURN(
          pass1_tmp_nzeros[thread],
          Image3I::Create(memory_manager, kGroupDimInBlocks,
                          kGroupDimInBlocks));
    }
    const Rect grect = frame_dim.BlockGroupRect(group_index);
    const int32_t* JXL_RESTRICT ac_rows[3] = {
        enc_state->coeffs[0]->PlaneRow(0, group_index, 0).ptr32,
        enc_state->coeffs[0]->PlaneRow(1, group_index, 0).ptr32,
        enc_state->coeffs[0]->PlaneRow(2, group_index, 0).ptr32,
    };
    JXL_RETURN_IF_ERROR(TokenizeCoefficients(
        token_cost_model->coeff_orders.data(), grect, ac_rows, ac_strategy,
        frame_header.chroma_subsampling, &pass1_tmp_nzeros[thread],
        &pass1_tokens[group_index], shared.quant_dc, raw_quant_field,
        block_ctx_map));
    for (size_t c = 0; c < 3; ++c) {
      for (size_t by = 0; by < grect.ysize(); ++by) {
        const int32_t* JXL_RESTRICT src_r =
            pass1_tmp_nzeros[thread].ConstPlaneRow(c, by);
        int32_t* JXL_RESTRICT dst_r =
            token_cost_model->pass1_nzeros.PlaneRow(c, grect.y0() + by) +
            grect.x0();
        memcpy(dst_r, src_r, grect.xsize() * sizeof(int32_t));
      }
    }
    return true;
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, frame_dim.num_groups, tok_init,
                                tok_group, "Pass1 Tokenize"));

  HistogramParams hist_params(cparams.speed_tier,
                              block_ctx_map.NumACContexts());
  if (cparams.speed_tier > SpeedTier::kTortoise) {
    hist_params.lz77_method = HistogramParams::LZ77Method::kNone;
  }
  if (cparams.decoding_speed_tier >= 1) {
    hist_params.max_histograms = 6;
  }
  EntropyEncodingData pass1_codes;
  JXL_ASSIGN_OR_RETURN(
      size_t pass1_ac_bits,
      BuildAndEncodeHistograms(memory_manager, hist_params,
                               block_ctx_map.NumACContexts(), pass1_tokens,
                               &pass1_codes, /*writer=*/nullptr, LayerType::Ac,
                               /*aux_out=*/nullptr));
  (void)pass1_ac_bits;

  token_cost_model->block_ctx_map = block_ctx_map;
  token_cost_model->cluster_of_ctx = pass1_codes.context_map;
  token_cost_model->uint_config = pass1_codes.uint_config;
  const size_t num_clusters = pass1_codes.encoding_info.size();
  token_cost_model->tok_bits.assign(
      num_clusters * AcTokenCostModel::kMaxTokenAlphabet, 12.5f);
  token_cost_model->small_val_bits.resize(
      num_clusters * AcTokenCostModel::kSmallValLimit);
  for (size_t cl = 0; cl < num_clusters; ++cl) {
    const auto& info_vec = pass1_codes.encoding_info[cl];
    const float unseen_bits = pass1_codes.use_prefix_code ? 14.0f : 12.5f;
    for (size_t tok = 0; tok < AcTokenCostModel::kMaxTokenAlphabet; ++tok) {
      float b = unseen_bits;
      if (tok < info_vec.size()) {
        if (pass1_codes.use_prefix_code) {
          if (info_vec.size() <= 1) {
            b = 0.0f;
          } else if (info_vec[tok].depth > 0) {
            b = static_cast<float>(info_vec[tok].depth);
          }
        } else if (info_vec[tok].freq_ > 0) {
          b = 12.0f - std::log2(static_cast<float>(info_vec[tok].freq_));
        }
      }
      token_cost_model->tok_bits[cl * AcTokenCostModel::kMaxTokenAlphabet +
                                 tok] = b;
    }
    for (uint32_t val = 0; val < AcTokenCostModel::kSmallValLimit; ++val) {
      token_cost_model
          ->small_val_bits[cl * AcTokenCostModel::kSmallValLimit + val] =
          token_cost_model->LargeValBits(cl, val);
    }
  }

  std::vector<size_t> strat_count(27, 0);
  std::vector<size_t> qf_res_count(512, 0);
  size_t total_first_blocks = 0;
  for (size_t by = 0; by < frame_dim.ysize_blocks; ++by) {
    AcStrategyRow acs_row = ac_strategy.ConstRow(by);
    const int32_t* qf_row = raw_quant_field.ConstRow(by);
    const int32_t* qf_top =
        (by > 0) ? raw_quant_field.ConstRow(by - 1) : nullptr;
    for (size_t bx = 0; bx < frame_dim.xsize_blocks; ++bx) {
      AcStrategy acs = acs_row[bx];
      if (!acs.IsFirstBlock()) continue;
      const uint8_t st = acs.RawStrategy();
      if (st < 27) strat_count[st]++;
      int32_t pred_qf = qf_row[bx];
      if (bx > 0 && by > 0) {
        pred_qf = (qf_row[bx - 1] + qf_top[bx] + 1) >> 1;
      } else if (bx > 0) {
        pred_qf = qf_row[bx - 1];
      } else if (by > 0) {
        pred_qf = qf_top[bx];
      }
      const int32_t res_idx =
          Clamp1<int32_t>(qf_row[bx] - pred_qf + 256, 0, 511);
      qf_res_count[res_idx]++;
      total_first_blocks++;
    }
  }
  const float fb_denom = static_cast<float>(total_first_blocks) + 13.5f;
  for (size_t st = 0; st < 27; ++st) {
    token_cost_model->acs_strategy_bits[st] = -std::log2(
        (static_cast<float>(strat_count[st]) + 0.5f) / fb_denom);
  }
  double qf_bits_sum = 0.0;
  if (total_first_blocks > 0) {
    const double inv_fb = 1.0 / static_cast<double>(total_first_blocks);
    for (size_t i = 0; i < 512; ++i) {
      if (qf_res_count[i] > 0) {
        const double p = static_cast<double>(qf_res_count[i]) * inv_fb;
        qf_bits_sum -= p * std::log2(p);
      }
    }
  }
  token_cost_model->qf_block_bits =
      Clamp1(static_cast<float>(qf_bits_sum), 1.0f, 3.8f);

  const size_t num_ctxs = block_ctx_map.num_ctxs;
  token_cost_model->rdo_nzero_bits.resize(num_ctxs * kNonZeroBuckets * 65);
  for (size_t nzero_ctx = 0; nzero_ctx < num_ctxs * kNonZeroBuckets;
       ++nzero_ctx) {
    const uint8_t cl = token_cost_model->cluster_of_ctx[nzero_ctx];
    for (uint32_t nz = 0; nz <= 64; ++nz) {
      token_cost_model->rdo_nzero_bits[nzero_ctx * 65 + nz] =
          token_cost_model->ValBits(cl, nz);
    }
  }

  token_cost_model->rdo_c_one.assign(num_ctxs * 32, 2.0f);
  token_cost_model->rdo_c_slope.assign(num_ctxs * 32, 1.5f);
  static constexpr size_t kNzSample[5] = {1, 2, 4, 8, 16};
  static constexpr float kNzWeight[5] = {0.30f, 0.25f, 0.20f, 0.15f, 0.10f};
  static constexpr float kPrevWeight[2] = {0.60f, 0.40f};
  for (size_t bctx = 0; bctx < num_ctxs; ++bctx) {
    const size_t z_off = block_ctx_map.ZeroDensityContextsOffset(bctx);
    for (size_t kf = 0; kf < 31; ++kf) {
      float b0_avg = 0.0f;
      float b1_avg = 0.0f;
      float b4_avg = 0.0f;
      for (size_t ni = 0; ni < 5; ++ni) {
        const size_t nz_ctx_off = kCoeffNumNonzeroContext[kNzSample[ni]];
        for (size_t prev = 0; prev < 2; ++prev) {
          const float w = kNzWeight[ni] * kPrevWeight[prev];
          const size_t ctx = z_off + (nz_ctx_off + kf) * 2 + prev;
          const uint8_t cl = token_cost_model->cluster_of_ctx[ctx];
          b0_avg += w * token_cost_model->ValBits(cl, 0);
          b1_avg += w * 0.5f *
                    (token_cost_model->ValBits(cl, 1) +
                     token_cost_model->ValBits(cl, 2));
          b4_avg += w * 0.5f *
                    (token_cost_model->ValBits(cl, 7) +
                     token_cost_model->ValBits(cl, 8));
        }
      }
      const float c_one =
          Clamp1((b1_avg - b0_avg) +
                     (0.25f + 0.035f * static_cast<float>(kf)) * b0_avg,
                 0.85f, 6.0f);
      const float c_slope = Clamp1(b4_avg - b1_avg, 0.65f, 4.5f);
      token_cost_model->rdo_c_one[bctx * 32 + kf] = c_one;
      token_cost_model->rdo_c_slope[bctx * 32 + kf] = c_slope;
    }
  }
  static const float kBias = 0.13731742964354549f;
  const float ratio = (cparams.butteraugli_distance + kBias) / (1.0f + kBias);
  token_cost_model->bits_to_cost_mul =
      0.428f * cost_delta * std::pow(ratio, 0.14f);
  block_ctx_map = BlockCtxMap();
  return true;
}

Status BuildPass2FrozenTokenCostModel(const FrameHeader& frame_header,
                                      const Image3F& opsin, const Rect& rect,
                                      ThreadPool* pool,
                                      PassesEncoderState* enc_state,
                                      bool pilot_coeffs_ready) {
  if (enc_state->token_cost_model == nullptr || enc_state->streaming_mode ||
      !enc_state->initialize_global_state ||
      enc_state->progressive_splitter.GetNumPasses() != 1) {
    return true;
  }
  const CompressParams& cparams = enc_state->cparams;
  PassesSharedState& shared = enc_state->shared;
  const FrameDimensions& frame_dim = shared.frame_dim;
  JxlMemoryManager* memory_manager = enc_state->memory_manager();

  // Detach Pass-1 token_cost_model so pilot ComputeCoefficients runs fast
  // deadzone quantization + backward EOB Viterbi (unbiased by Pass-1 prices).
  std::unique_ptr<AcTokenCostModel> frozen_tcm =
      std::move(enc_state->token_cost_model);

  if (!pilot_coeffs_ready) {
    JXL_ASSIGN_OR_RETURN(
        Image3F pilot_dc,
        Image3F::Create(memory_manager, frame_dim.xsize_blocks,
                        frame_dim.ysize_blocks));
    const auto compute_pilot_coeffs = [&](size_t group_idx,
                                          size_t /*thread*/) -> Status {
      return ComputeCoefficients(group_idx, enc_state, opsin, rect, &pilot_dc);
    };
    JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, frame_dim.num_groups,
                                  ThreadPool::NoInit, compute_pilot_coeffs,
                                  "Pass2A PilotCoeffs"));
  }

  // Compute exact Pass-2 coefficient scan orders on pilot coeffs.
  frozen_tcm->coeff_orders.resize(kCoeffOrderMaxSize);
  auto used_orders_info = ComputeUsedOrders(
      cparams.speed_tier, shared.ac_strategy, Rect(shared.raw_quant_field));
  uint32_t pilot_used_orders = 0;
  constexpr uint32_t kAllOrdersMask = (1u << kNumOrders) - 1u;
  JXL_RETURN_IF_ERROR(ComputeCoeffOrder(
      cparams.speed_tier, *enc_state->coeffs[0], shared.ac_strategy, frame_dim,
      pilot_used_orders, /*prev_used_acs=*/0, kAllOrdersMask,
      used_orders_info.second, frozen_tcm->coeff_orders.data()));

  // Refresh Pass-2 strategy and qf residual bit prices from Pass-2 layout.
  {
    size_t strat_count[27] = {};
    size_t qf_res_count[512] = {};
    size_t total_first_blocks = 0;
    for (size_t by = 0; by < frame_dim.ysize_blocks; ++by) {
      AcStrategyRow acs_row = shared.ac_strategy.ConstRow(by);
      const int32_t* qf_row = shared.raw_quant_field.ConstRow(by);
      const int32_t* qf_top =
          (by > 0) ? shared.raw_quant_field.ConstRow(by - 1) : nullptr;
      for (size_t bx = 0; bx < frame_dim.xsize_blocks; ++bx) {
        AcStrategy acs = acs_row[bx];
        if (!acs.IsFirstBlock()) continue;
        const uint8_t st = acs.RawStrategy();
        if (st < 27) strat_count[st]++;
        int32_t pred_qf = qf_row[bx];
        if (bx > 0 && by > 0) {
          pred_qf = (qf_row[bx - 1] + qf_top[bx] + 1) >> 1;
        } else if (bx > 0) {
          pred_qf = qf_row[bx - 1];
        } else if (by > 0) {
          pred_qf = qf_top[bx];
        }
        const int32_t res_idx =
            Clamp1<int32_t>(qf_row[bx] - pred_qf + 256, 0, 511);
        qf_res_count[res_idx]++;
        total_first_blocks++;
      }
    }
    const float fb_denom = static_cast<float>(total_first_blocks) + 13.5f;
    for (size_t st = 0; st < 27; ++st) {
      frozen_tcm->acs_strategy_bits[st] = -std::log2(
          (static_cast<float>(strat_count[st]) + 0.5f) / fb_denom);
    }
    double qf_bits_sum = 0.0;
    if (total_first_blocks > 0) {
      const double inv_fb = 1.0 / static_cast<double>(total_first_blocks);
      for (size_t i = 0; i < 512; ++i) {
        if (qf_res_count[i] > 0) {
          const double p = static_cast<double>(qf_res_count[i]) * inv_fb;
          qf_bits_sum -= p * std::log2(p);
        }
      }
    }
    frozen_tcm->qf_block_bits =
        Clamp1(static_cast<float>(qf_bits_sum), 1.0f, 3.8f);
  }

  // Compute exact Pass-2 block context map (1D DP qf_thresholds + JS-div).
  frozen_tcm->block_ctx_map = BlockCtxMap();
  if (cparams.speed_tier < SpeedTier::kFalcon) {
    JXL_RETURN_IF_ERROR(FindBestBlockEntropyModelFromCoeffs(
        frame_header, frozen_tcm->coeff_orders.data(), enc_state,
        &frozen_tcm->block_ctx_map));
  } else {
    FindBestBlockEntropyModel(cparams, shared.raw_quant_field,
                              shared.ac_strategy, &frozen_tcm->block_ctx_map);
  }

  // Tokenize pilot coefficients on exact Pass-2 orders and block_ctx_map.
  if (!pilot_coeffs_ready) {
    ZeroFillImage(&shared.quant_dc);
  }
  if (frozen_tcm->pass1_nzeros.xsize() == 0) {
    JXL_ASSIGN_OR_RETURN(
        frozen_tcm->pass1_nzeros,
        Image3I::Create(memory_manager, frame_dim.xsize_blocks,
                        frame_dim.ysize_blocks));
  }
  std::vector<std::vector<Token>> pilot_tokens(frame_dim.num_groups);
  std::vector<Image3I> pilot_tmp_nzeros;
  const auto tok_init = [&](const size_t num_threads) -> Status {
    pilot_tmp_nzeros.resize(num_threads);
    return true;
  };
  const auto tok_group = [&](const uint32_t group_index,
                             const size_t thread) -> Status {
    if (pilot_tmp_nzeros[thread].xsize() == 0) {
      JXL_ASSIGN_OR_RETURN(
          pilot_tmp_nzeros[thread],
          Image3I::Create(memory_manager, kGroupDimInBlocks,
                          kGroupDimInBlocks));
    }
    const Rect grect = frame_dim.BlockGroupRect(group_index);
    const int32_t* JXL_RESTRICT ac_rows[3] = {
        enc_state->coeffs[0]->PlaneRow(0, group_index, 0).ptr32,
        enc_state->coeffs[0]->PlaneRow(1, group_index, 0).ptr32,
        enc_state->coeffs[0]->PlaneRow(2, group_index, 0).ptr32,
    };
    JXL_RETURN_IF_ERROR(TokenizeCoefficients(
        frozen_tcm->coeff_orders.data(), grect, ac_rows, shared.ac_strategy,
        frame_header.chroma_subsampling, &pilot_tmp_nzeros[thread],
        &pilot_tokens[group_index], shared.quant_dc, shared.raw_quant_field,
        frozen_tcm->block_ctx_map));
    for (size_t c = 0; c < 3; ++c) {
      for (size_t by = 0; by < grect.ysize(); ++by) {
        const int32_t* JXL_RESTRICT src_r =
            pilot_tmp_nzeros[thread].ConstPlaneRow(c, by);
        int32_t* JXL_RESTRICT dst_r =
            frozen_tcm->pass1_nzeros.PlaneRow(c, grect.y0() + by) + grect.x0();
        memcpy(dst_r, src_r, grect.xsize() * sizeof(int32_t));
      }
    }
    return true;
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, frame_dim.num_groups, tok_init,
                                tok_group, "Pass2A PilotTokenize"));

  // Cluster histograms with kBest + kPrecise ANS, keeping kNone uint_method
  // (default HybridUintConfig(4,2,0)) to avoid single-token config flips.
  HistogramParams hist_params(cparams.speed_tier,
                              frozen_tcm->block_ctx_map.NumACContexts());
  if (cparams.speed_tier <= SpeedTier::kSquirrel) {
    hist_params.clustering = HistogramParams::ClusteringType::kBest;
    hist_params.uint_method = HistogramParams::HybridUintMethod::kNone;
    hist_params.ans_histogram_strategy =
        HistogramParams::ANSHistogramStrategy::kPrecise;
  }
  if (cparams.speed_tier > SpeedTier::kTortoise) {
    hist_params.lz77_method = HistogramParams::LZ77Method::kNone;
  }
  if (cparams.decoding_speed_tier >= 1) {
    hist_params.max_histograms = 6;
  }
  EntropyEncodingData pilot_codes;
  JXL_ASSIGN_OR_RETURN(
      size_t pilot_ac_bits,
      BuildAndEncodeHistograms(
          memory_manager, hist_params,
          frozen_tcm->block_ctx_map.NumACContexts(), pilot_tokens, &pilot_codes,
          /*writer=*/nullptr, LayerType::Ac, /*aux_out=*/nullptr));
  (void)pilot_ac_bits;

  frozen_tcm->cluster_of_ctx = pilot_codes.context_map;
  frozen_tcm->uint_config = pilot_codes.uint_config;
  const size_t num_clusters = pilot_codes.encoding_info.size();
  frozen_tcm->tok_bits.assign(
      num_clusters * AcTokenCostModel::kMaxTokenAlphabet, 12.5f);
  frozen_tcm->small_val_bits.resize(
      num_clusters * AcTokenCostModel::kSmallValLimit);
  for (size_t cl = 0; cl < num_clusters; ++cl) {
    const auto& info_vec = pilot_codes.encoding_info[cl];
    const float unseen_bits = pilot_codes.use_prefix_code ? 14.0f : 12.5f;
    for (size_t tok = 0; tok < AcTokenCostModel::kMaxTokenAlphabet; ++tok) {
      float b = unseen_bits;
      if (tok < info_vec.size()) {
        if (pilot_codes.use_prefix_code) {
          if (info_vec.size() <= 1) {
            b = 0.0f;
          } else if (info_vec[tok].depth > 0) {
            b = static_cast<float>(info_vec[tok].depth);
          }
        } else if (info_vec[tok].freq_ > 0) {
          b = 12.0f - std::log2(static_cast<float>(info_vec[tok].freq_));
        }
      }
      frozen_tcm->tok_bits[cl * AcTokenCostModel::kMaxTokenAlphabet + tok] = b;
    }
    for (uint32_t val = 0; val < AcTokenCostModel::kSmallValLimit; ++val) {
      frozen_tcm->small_val_bits[cl * AcTokenCostModel::kSmallValLimit + val] =
          frozen_tcm->LargeValBits(cl, val);
    }
  }

  enc_state->token_cost_model = std::move(frozen_tcm);
  return true;
}

Status LossyFrameHeuristics(const FrameHeader& frame_header,
                            PassesEncoderState* enc_state,
                            ModularFrameEncoder* modular_frame_encoder,
                            const Image3F* linear, Image3F* opsin,
                            const Rect& rect, const JxlCmsInterface& cms,
                            ThreadPool* pool, AuxOut* aux_out) {
  const CompressParams& cparams = enc_state->cparams;
  const bool streaming_mode = enc_state->streaming_mode;
  const bool initialize_global_state = enc_state->initialize_global_state;
  PassesSharedState& shared = enc_state->shared;
  const FrameDimensions& frame_dim = shared.frame_dim;
  ImageFeatures& image_features = shared.image_features;
  DequantMatrices& matrices = shared.matrices;
  Quantizer& quantizer = shared.quantizer;
  ImageF& initial_quant_masking1x1 = enc_state->initial_quant_masking1x1;
  ImageI& raw_quant_field = shared.raw_quant_field;
  ColorCorrelationMap& cmap = shared.cmap;
  AcStrategyImage& ac_strategy = shared.ac_strategy;
  BlockCtxMap& block_ctx_map = shared.block_ctx_map;
  JxlMemoryManager* memory_manager = enc_state->memory_manager();
  enc_state->token_cost_model.reset();

  // Find and subtract splines.
  bool override_splines = cparams.custom_splines.HasAny();
  if (override_splines) {
    image_features.splines.SetData(cparams.custom_splines);
  }
  if (!streaming_mode && cparams.speed_tier <= SpeedTier::kSquirrel) {
    if (!override_splines) {
      image_features.splines = FindSplines(*opsin);
    }
    JXL_RETURN_IF_ERROR(image_features.splines.InitializeDrawCache(
        opsin->xsize(), opsin->ysize(), cmap.base()));
    image_features.splines.SubtractFrom(opsin);
  }

  // Find and subtract patches/dots.
  if (!streaming_mode &&
      ApplyOverride(cparams.patches,
                    cparams.speed_tier <= SpeedTier::kSquirrel)) {
    JXL_RETURN_IF_ERROR(
        FindBestPatchDictionary(*opsin, enc_state, cms, pool, aux_out));
    JXL_RETURN_IF_ERROR(
        PatchDictionaryEncoder::SubtractFrom(image_features.patches, opsin));
  }

  const float quant_dc = InitialQuantDC(cparams.butteraugli_distance);

  // TODO(veluca): we can now run all the code from here to FindBestQuantizer
  // (excluded) one rect at a time. Do that.

  // Dependency graph:
  //
  // input: either XYB or input image
  //
  // input image -> XYB [optional]
  // XYB -> initial quant field
  // XYB -> Gaborished XYB
  // Gaborished XYB -> CfL1
  // initial quant field, Gaborished XYB, CfL1 -> ACS
  // initial quant field, ACS, Gaborished XYB -> EPF control field
  // initial quant field -> adjusted initial quant field
  // adjusted initial quant field, ACS -> raw quant field
  // raw quant field, ACS, Gaborished XYB -> CfL2
  //
  // output: Gaborished XYB, CfL, ACS, raw quant field, EPF control field.

  AcStrategyHeuristics acs_heuristics(memory_manager, cparams);
  CfLHeuristics cfl_heuristics(memory_manager);
  ImageF initial_quant_field;
  const bool enable_rdo =
      cparams.speed_tier <= SpeedTier::kSquirrel &&
      !cparams.disable_perceptual_optimizations &&
      !shared.metadata->m.color_encoding.IsGray() &&
      cparams.butteraugli_distance >= 0.35f &&
      cparams.butteraugli_distance <= 15.0f && cparams.resampling <= 1 &&
      cparams.ec_resampling <= 1 && cparams.gaborish != Override::kOff &&
      cparams.epf != 3 &&
      enc_state->progressive_splitter.GetNumPasses() == 1 &&
      frame_dim.xsize_blocks * frame_dim.ysize_blocks > 64;
  if (!enable_rdo) {
    enc_state->band_qf[0][0] = ImageF();
    if (cparams.speed_tier <= SpeedTier::kHare) {
      JXL_ASSIGN_OR_RETURN(enc_state->band_qf[1][0],
                           ImageF::Create(memory_manager, frame_dim.xsize_blocks,
                                          frame_dim.ysize_blocks));
      FillImage(1.0f, &enc_state->band_qf[1][0]);
    } else {
      enc_state->band_qf[1][0] = ImageF();
    }
  }

  // Compute an initial estimate of the quantization field.
  // Call InitialQuantField only in Hare mode or slower. Otherwise, rely
  // on simple heuristics in FindBestAcStrategy, or set a constant for Falcon
  // mode.
  if (cparams.speed_tier > SpeedTier::kHare ||
      cparams.disable_perceptual_optimizations) {
    JXL_ASSIGN_OR_RETURN(initial_quant_field,
                         ImageF::Create(memory_manager, frame_dim.xsize_blocks,
                                        frame_dim.ysize_blocks));
    float q = 0.79 / cparams.butteraugli_distance;
    FillImage(q, &initial_quant_field);
    if (cparams.disable_perceptual_optimizations) {
      float masking = 1.0f / (q + 0.001f);
      JXL_ASSIGN_OR_RETURN(
          initial_quant_masking1x1,
          ImageF::Create(memory_manager, frame_dim.xsize, frame_dim.ysize));
      FillImage(masking, &initial_quant_masking1x1);
    }
    quantizer.ComputeGlobalScaleAndQuant(quant_dc, q, 0);
  } else {
    // Call this here, as it relies on pre-gaborish values.
    float butteraugli_distance_for_iqf = cparams.butteraugli_distance;
    if (!frame_header.loop_filter.gab) {
      butteraugli_distance_for_iqf *= 0.62f;
    }
    JXL_ASSIGN_OR_RETURN(
        initial_quant_field,
        InitialQuantField(butteraugli_distance_for_iqf, *opsin, rect, pool,
                          1.0f, &initial_quant_masking1x1,
                          enable_rdo ? enc_state->band_qf : nullptr,
                          (frame_header.flags & FrameHeader::kNoise) != 0));
    float q = 0.39 / cparams.butteraugli_distance;
    quantizer.ComputeGlobalScaleAndQuant(quant_dc, q, 0);
  }

  // TODO(veluca): do something about animations.

  // Apply inverse-gaborish.
  if (frame_header.loop_filter.gab) {
    // Changing the weight here to 0.99f would help to reduce ringing in
    // generation loss.
    const float gab_w =
        enable_rdo
            ? (0.988f + 0.006f * Clamp1((cparams.butteraugli_distance - 2.5f) /
                                            5.0f,
                                        0.0f, 1.0f))
            : 1.0f;
    float weight[3] = {
        gab_w,
        gab_w,
        gab_w,
    };
    JXL_RETURN_IF_ERROR(GaborishInverse(opsin, rect, weight, pool));
  }

  if (initialize_global_state) {
    JXL_RETURN_IF_ERROR(FindBestDequantMatrices(
        memory_manager, cparams, frame_header, frame_dim, enable_rdo,
        initial_quant_field, /*band_qf=*/nullptr, modular_frame_encoder,
        &matrices));
    quantizer.RecomputeFromGlobalScale();
  }

  enc_state->x_qm_multiplier =
      std::pow(1.25f, frame_header.x_qm_scale - 2.0f);
  enc_state->b_qm_multiplier =
      std::pow(1.25f, frame_header.b_qm_scale - 2.0f);

  JXL_RETURN_IF_ERROR(cfl_heuristics.Init(rect));
  JXL_RETURN_IF_ERROR(acs_heuristics.Init(
      *opsin, rect, &initial_quant_field, initial_quant_masking1x1, &matrices,
      (enable_rdo || cparams.speed_tier <= SpeedTier::kHare)
          ? enc_state->band_qf
          : nullptr,
      enc_state->x_qm_multiplier, enc_state->b_qm_multiplier, quantizer.Scale(),
      quantizer.InvGlobalScale()));

  const bool use_two_pass_rdo =
      enable_rdo && !streaming_mode && cparams.use_full_image_heuristics;
  ImageF initial_quant_field_base;
  std::vector<ImageF> band_qf_base;
  if (use_two_pass_rdo) {
    band_qf_base.resize(12);
    JXL_ASSIGN_OR_RETURN(
        initial_quant_field_base,
        ImageF::Create(memory_manager, initial_quant_field.xsize(),
                       initial_quant_field.ysize()));
    JXL_RETURN_IF_ERROR(
        CopyImageTo(initial_quant_field, &initial_quant_field_base));
    for (size_t c = 0; c < 3; ++c) {
      for (size_t s = 0; s < 4; ++s) {
        if (enc_state->band_qf[c][s].xsize() > 0 &&
            enc_state->band_qf[c][s].ysize() > 0) {
          JXL_ASSIGN_OR_RETURN(
              band_qf_base[c * 4 + s],
              ImageF::Create(memory_manager, enc_state->band_qf[c][s].xsize(),
                             enc_state->band_qf[c][s].ysize()));
          JXL_RETURN_IF_ERROR(
              CopyImageTo(enc_state->band_qf[c][s], &band_qf_base[c * 4 + s]));
        }
      }
    }
  }

  auto restore_tile_quant = [&](const Rect& r) -> Status {
    JXL_RETURN_IF_ERROR(
        CopyImageTo(r, initial_quant_field_base, r, &initial_quant_field));
    for (size_t c = 0; c < 3; ++c) {
      for (size_t s = 0; s < 4; ++s) {
        if (band_qf_base[c * 4 + s].xsize() > 0 &&
            band_qf_base[c * 4 + s].ysize() > 0) {
          JXL_RETURN_IF_ERROR(CopyImageTo(r, band_qf_base[c * 4 + s], r,
                                          &enc_state->band_qf[c][s]));
        }
      }
    }
    return true;
  };

  // Schedule independent 256x256px AC groups in parallel, and iterate the 4x4
  // 64x64px CfL tiles inside each AC group in causal row-major order so
  // ClampedGradient CfL prediction is 100% deterministic and race-free while
  // preserving full arbitrary-origin block placement inside each AC group.
  const size_t acs_tile_dim_blocks =
      enable_rdo ? kGroupDimInBlocks : kEncTileDimInBlocks;
  const size_t n_groups_x =
      DivCeil(frame_dim.xsize_blocks, acs_tile_dim_blocks);
  const size_t num_groups =
      n_groups_x * DivCeil(frame_dim.ysize_blocks, acs_tile_dim_blocks);
  auto process_tile = [&](const uint32_t gid, const size_t thread) -> Status {
    const size_t gx = gid % n_groups_x;
    const size_t gy = gid / n_groups_x;
    const size_t gby0 = gy * acs_tile_dim_blocks;
    const size_t gby1 =
        std::min((gy + 1) * acs_tile_dim_blocks, frame_dim.ysize_blocks);
    const size_t gbx0 = gx * acs_tile_dim_blocks;
    const size_t gbx1 =
        std::min((gx + 1) * acs_tile_dim_blocks, frame_dim.xsize_blocks);
    const Rect group_rect(gbx0, gby0, gbx1 - gbx0, gby1 - gby0);

    if (cparams.speed_tier <= SpeedTier::kSquirrel) {
      if (enable_rdo) {
        quantizer.SetQuantFieldRect(initial_quant_field, group_rect,
                                    &raw_quant_field);
      }
      for (size_t by0 = gby0; by0 < gby1; by0 += kEncTileDimInBlocks) {
        const size_t by1 = std::min(by0 + kEncTileDimInBlocks, gby1);
        for (size_t bx0 = gbx0; bx0 < gbx1; bx0 += kEncTileDimInBlocks) {
          const size_t bx1 = std::min(bx0 + kEncTileDimInBlocks, gbx1);
          const Rect r(bx0, by0, bx1 - bx0, by1 - by0);
          JXL_RETURN_IF_ERROR(cfl_heuristics.ComputeTile(
              r, *opsin, rect, matrices,
              /*ac_strategy=*/nullptr,
              /*raw_quant_field=*/enable_rdo ? &raw_quant_field : nullptr,
              /*quantizer=*/enable_rdo ? &quantizer : nullptr, /*fast=*/false,
              thread, &cmap,
              /*shrink_towards_zero=*/!enable_rdo ||
                  cparams.noise == Override::kOn,
              enc_state->x_qm_multiplier, enc_state->b_qm_multiplier,
              cparams.butteraugli_distance));
        }
      }
    }

    JXL_RETURN_IF_ERROR(
        acs_heuristics.ProcessRect(group_rect, cmap, &ac_strategy, thread));
    quantizer.SetQuantFieldRect(initial_quant_field, group_rect,
                                &raw_quant_field);

    if (cparams.speed_tier <= SpeedTier::kHare) {
      for (size_t by0 = gby0; by0 < gby1; by0 += kEncTileDimInBlocks) {
        const size_t by1 = std::min(by0 + kEncTileDimInBlocks, gby1);
        for (size_t bx0 = gbx0; bx0 < gbx1; bx0 += kEncTileDimInBlocks) {
          const size_t bx1 = std::min(bx0 + kEncTileDimInBlocks, gbx1);
          const Rect r(bx0, by0, bx1 - bx0, by1 - by0);
          JXL_RETURN_IF_ERROR(cfl_heuristics.ComputeTile(
              r, *opsin, rect, matrices, &ac_strategy, &raw_quant_field,
              &quantizer, /*fast=*/cparams.speed_tier >= SpeedTier::kWombat,
              thread, &cmap,
              /*shrink_towards_zero=*/!enable_rdo ||
                  cparams.noise == Override::kOn,
              enc_state->x_qm_multiplier, enc_state->b_qm_multiplier,
              cparams.butteraugli_distance));
        }
      }
    }
    return true;
  };

  const auto prepare = [&](const size_t num_threads) -> Status {
    JXL_RETURN_IF_ERROR(acs_heuristics.PrepareForThreads(num_threads));
    JXL_RETURN_IF_ERROR(cfl_heuristics.PrepareForThreads(num_threads));
    return true;
  };
  JXL_RETURN_IF_ERROR(
      RunOnPool(pool, 0, num_groups, prepare, process_tile, "Enc Heuristics"));

  std::unique_ptr<AcTokenCostModel> token_cost_model;
  if (use_two_pass_rdo) {
    if (initialize_global_state && cparams.butteraugli_distance > 1.05f) {
      JXL_RETURN_IF_ERROR(FindBestDequantMatrices(
          memory_manager, cparams, frame_header, frame_dim, enable_rdo,
          initial_quant_field, enc_state->band_qf, modular_frame_encoder,
          &matrices));
      quantizer.RecomputeFromGlobalScale();
      const uint32_t all_acs_64_mask =
          (1u << static_cast<uint32_t>(AcStrategyType::DCT128X128)) - 1u;
      JXL_RETURN_IF_ERROR(
          matrices.EnsureComputed(memory_manager, all_acs_64_mask));
      for (size_t c = 0; c < 3; ++c) {
        const float* dct8_matrix = matrices.Matrix(AcStrategyType::DCT, c);
        const float m_ref = dct8_matrix[1];
        constexpr float kBetaCsf = 0.08f;
        float csf_sum = 0.0f;
        for (size_t i = 0; i < 64; ++i) {
          const float m_uv =
              (i == 0) ? m_ref : std::max(dct8_matrix[i], m_ref * 0.5f);
          const float w_uv = std::pow(m_ref / m_uv, kBetaCsf);
          acs_heuristics.config.csf_w[c][i] = w_uv;
          csf_sum += w_uv;
        }
        const float csf_norm = 61.8f / csf_sum;
        for (size_t i = 0; i < 64; ++i) {
          acs_heuristics.config.csf_w[c][i] *= csf_norm;
        }
      }
    }

    token_cost_model = jxl::make_unique<AcTokenCostModel>();
    JXL_RETURN_IF_ERROR(BuildPass1TokenCostModel(
        frame_header, enc_state, *opsin, rect, acs_heuristics.config.cost_delta,
        pool, token_cost_model.get()));

    JXL_RETURN_IF_ERROR(
        CopyImageTo(initial_quant_field_base, &initial_quant_field));
    for (size_t c = 0; c < 3; ++c) {
      for (size_t s = 0; s < 4; ++s) {
        if (band_qf_base[c * 4 + s].xsize() > 0 &&
            band_qf_base[c * 4 + s].ysize() > 0) {
          JXL_RETURN_IF_ERROR(
              CopyImageTo(band_qf_base[c * 4 + s], &enc_state->band_qf[c][s]));
        }
      }
    }
    acs_heuristics.config.token_cost_model = token_cost_model.get();
    uint32_t disabled_tables_mask = 0;
    if (initialize_global_state) {
      size_t table_instances[kNumQuantTables] = {};
      size_t table_blocks[kNumQuantTables] = {};
      for (size_t y = 0; y < ac_strategy.ysize(); ++y) {
        AcStrategyRow row = ac_strategy.ConstRow(y);
        for (size_t x = 0; x < ac_strategy.xsize(); ++x) {
          AcStrategy acs = row[x];
          if (!acs.IsFirstBlock()) continue;
          const size_t st = static_cast<size_t>(acs.Strategy());
          const size_t t_idx = static_cast<size_t>(kAcStrategyToQuantTableMap[st]);
          table_instances[t_idx]++;
          table_blocks[t_idx] += acs.covered_blocks_x() * acs.covered_blocks_y();
        }
      }
      const auto& encs = matrices.encodings();
      const float d_prune = std::max(0.35f, cparams.butteraugli_distance);
      for (size_t t = 1; t < kNumQuantTables && t < encs.size(); ++t) {
        if (encs[t].mode != QuantEncoding::kQuantModeDCT) continue;
        const float tax_bits =
            3.0f + 48.0f * static_cast<float>(encs[t].dct_params.num_distance_bands);
        const float est_gain =
            2.5f * static_cast<float>(table_instances[t]) +
            (3.2f / (0.5f + 0.6f * d_prune)) * static_cast<float>(table_blocks[t]);
        if (est_gain < 0.75f * tax_bits) {
          disabled_tables_mask |= (1u << t);
        }
      }
    }
    acs_heuristics.config.disabled_tables_mask = disabled_tables_mask;

    auto process_tile_pass2 = [&](const uint32_t gid,
                                  const size_t thread) -> Status {
      const size_t gx = gid % n_groups_x;
      const size_t gy = gid / n_groups_x;
      const size_t gby0 = gy * acs_tile_dim_blocks;
      const size_t gby1 =
          std::min((gy + 1) * acs_tile_dim_blocks, frame_dim.ysize_blocks);
      const size_t gbx0 = gx * acs_tile_dim_blocks;
      const size_t gbx1 =
          std::min((gx + 1) * acs_tile_dim_blocks, frame_dim.xsize_blocks);
      const Rect group_rect(gbx0, gby0, gbx1 - gbx0, gby1 - gby0);

      int32_t prev_x[4][4] = {};
      int32_t prev_b[4][4] = {};
      for (size_t by0 = gby0; by0 < gby1; by0 += kEncTileDimInBlocks) {
        const size_t ty = by0 / kEncTileDimInBlocks;
        const size_t ly = (by0 - gby0) / kEncTileDimInBlocks;
        for (size_t bx0 = gbx0; bx0 < gbx1; bx0 += kEncTileDimInBlocks) {
          const size_t tx = bx0 / kEncTileDimInBlocks;
          const size_t lx = (bx0 - gbx0) / kEncTileDimInBlocks;
          prev_x[ly][lx] = cmap.ytox_map.ConstRow(ty)[tx];
          prev_b[ly][lx] = cmap.ytob_map.ConstRow(ty)[tx];
        }
      }

      JXL_RETURN_IF_ERROR(
          acs_heuristics.ProcessRect(group_rect, cmap, &ac_strategy, thread));
      quantizer.SetQuantFieldRect(initial_quant_field, group_rect,
                                  &raw_quant_field);

      if (cparams.speed_tier <= SpeedTier::kHare) {
        int32_t max_cfl_delta = 0;
        for (size_t by0 = gby0; by0 < gby1; by0 += kEncTileDimInBlocks) {
          const size_t by1 = std::min(by0 + kEncTileDimInBlocks, gby1);
          const size_t ty = by0 / kEncTileDimInBlocks;
          const size_t ly = (by0 - gby0) / kEncTileDimInBlocks;
          for (size_t bx0 = gbx0; bx0 < gbx1; bx0 += kEncTileDimInBlocks) {
            const size_t bx1 = std::min(bx0 + kEncTileDimInBlocks, gbx1);
            const size_t tx = bx0 / kEncTileDimInBlocks;
            const size_t lx = (bx0 - gbx0) / kEncTileDimInBlocks;
            const Rect r(bx0, by0, bx1 - bx0, by1 - by0);
            JXL_RETURN_IF_ERROR(cfl_heuristics.ComputeTile(
                r, *opsin, rect, matrices, &ac_strategy, &raw_quant_field,
                &quantizer, /*fast=*/cparams.speed_tier >= SpeedTier::kWombat,
                thread, &cmap,
                /*shrink_towards_zero=*/cparams.noise == Override::kOn,
                enc_state->x_qm_multiplier, enc_state->b_qm_multiplier,
                cparams.butteraugli_distance));
            max_cfl_delta = std::max(
                max_cfl_delta,
                std::max(std::abs(int32_t{cmap.ytox_map.ConstRow(ty)[tx]} -
                                  prev_x[ly][lx]),
                         std::abs(int32_t{cmap.ytob_map.ConstRow(ty)[tx]} -
                                  prev_b[ly][lx])));
          }
        }
        if (cparams.speed_tier <= SpeedTier::kSquirrel && max_cfl_delta >= 5) {
          JXL_RETURN_IF_ERROR(restore_tile_quant(group_rect));
          JXL_RETURN_IF_ERROR(acs_heuristics.ProcessRect(
              group_rect, cmap, &ac_strategy, thread));
          quantizer.SetQuantFieldRect(initial_quant_field, group_rect,
                                      &raw_quant_field);
          for (size_t by0 = gby0; by0 < gby1; by0 += kEncTileDimInBlocks) {
            const size_t by1 = std::min(by0 + kEncTileDimInBlocks, gby1);
            for (size_t bx0 = gbx0; bx0 < gbx1; bx0 += kEncTileDimInBlocks) {
              const size_t bx1 = std::min(bx0 + kEncTileDimInBlocks, gbx1);
              const Rect r(bx0, by0, bx1 - bx0, by1 - by0);
              JXL_RETURN_IF_ERROR(cfl_heuristics.ComputeTile(
                  r, *opsin, rect, matrices, &ac_strategy, &raw_quant_field,
                  &quantizer, /*fast=*/false, thread, &cmap,
                  /*shrink_towards_zero=*/cparams.noise == Override::kOn,
                  enc_state->x_qm_multiplier, enc_state->b_qm_multiplier,
                  cparams.butteraugli_distance));
            }
          }
        }
      }
      return true;
    };
    JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, num_groups, prepare,
                                  process_tile_pass2, "Enc Heuristics P2"));
    acs_heuristics.config.token_cost_model = nullptr;
    acs_heuristics.config.disabled_tables_mask = 0;
  }

  JXL_RETURN_IF_ERROR(acs_heuristics.Finalize(frame_dim, ac_strategy, aux_out));

  if (!streaming_mode && initialize_global_state) {
    uint32_t used_tables_mask = 0;
    uint32_t used_acs_mask = 1u;
    for (size_t y = 0; y < ac_strategy.ysize(); ++y) {
      AcStrategyRow row = ac_strategy.ConstRow(y);
      for (size_t x = 0; x < ac_strategy.xsize(); ++x) {
        AcStrategy acs = row[x];
        if (!acs.IsFirstBlock()) continue;
        const size_t st = static_cast<size_t>(acs.Strategy());
        const size_t table_idx =
            static_cast<size_t>(kAcStrategyToQuantTableMap[st]);
        used_tables_mask |= (1u << table_idx);
        used_acs_mask |= (1u << st);
      }
    }
    if (!use_two_pass_rdo && enable_rdo &&
        cparams.butteraugli_distance > 1.05f) {
      JXL_RETURN_IF_ERROR(FindBestDequantMatrices(
          memory_manager, cparams, frame_header, frame_dim, enable_rdo,
          initial_quant_field, enc_state->band_qf, modular_frame_encoder,
          &matrices));
      quantizer.RecomputeFromGlobalScale();
    }
    const bool defer_prune =
        use_two_pass_rdo && enc_state->progressive_splitter.GetNumPasses() == 1;
    if (!defer_prune) {
      matrices.PruneUnusedEncodings(used_tables_mask);
    }
    if (enable_rdo) {
      const uint32_t ensure_mask =
          defer_prune
              ? ((1u << static_cast<uint32_t>(AcStrategyType::DCT128X128)) - 1u)
              : used_acs_mask;
      JXL_RETURN_IF_ERROR(
          matrices.EnsureComputed(memory_manager, ensure_mask));
    }
  }

  // Refine quantization levels.
  if (!streaming_mode && !cparams.disable_perceptual_optimizations) {
    ImageB& epf_sharpness = shared.epf_sharpness;
    FillPlane(static_cast<uint8_t>(4), &epf_sharpness, Rect(epf_sharpness));
    JXL_RETURN_IF_ERROR(FindBestQuantizer(frame_header, linear, *opsin,
                                          initial_quant_field, enc_state, cms,
                                          pool, aux_out));
  }
  enc_state->token_cost_model = std::move(token_cost_model);

  // Choose a context model that depends on the amount of quantization for AC.
  if (cparams.speed_tier < SpeedTier::kFalcon && initialize_global_state) {
    FindBestBlockEntropyModel(cparams, raw_quant_field, ac_strategy,
                              &block_ctx_map);
  }
  return true;
}

}  // namespace jxl
