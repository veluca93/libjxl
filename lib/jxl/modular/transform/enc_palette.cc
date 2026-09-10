// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/modular/transform/enc_palette.h"

#include <jxl/memory_manager.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <limits>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/pack_signed.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/options.h"
#include "lib/jxl/modular/transform/enc_transform.h"
#include "lib/jxl/modular/transform/palette.h"
#include "lib/jxl/modular/transform/transform.h"

namespace jxl {

namespace palette_internal {

static constexpr bool kEncodeToHighQualityImplicitPalette = true;

// Inclusive.
static constexpr int kMinImplicitPaletteIndex = -(2 * 72 - 1);

float ColorDistance(const std::vector<float> &JXL_RESTRICT a,
                    const std::vector<pixel_type> &JXL_RESTRICT b) {
  JXL_DASSERT(a.size() == b.size());
  float distance = 0;
  float ave3 = 0;
  if (a.size() >= 3) {
    ave3 = (a[0] + b[0] + a[1] + b[1] + a[2] + b[2]) * (1.21f / 3.0f);
  }
  float sum_a = 0;
  float sum_b = 0;
  for (size_t c = 0; c < a.size(); ++c) {
    const float difference =
        static_cast<float>(a[c]) - static_cast<float>(b[c]);
    float weight = c == 0 ? 3 : c == 1 ? 5 : 2;
    if (c < 3 && (a[c] + b[c] >= ave3)) {
      const float add_w[3] = {
          1.15,
          1.15,
          1.12,
      };
      weight += add_w[c];
      if (c == 2 && ((a[2] + b[2]) < 1.22 * ave3)) {
        weight -= 0.5;
      }
    }
    distance += difference * difference * weight * weight;
    const int sum_weight = c == 0 ? 3 : c == 1 ? 5 : 1;
    sum_a += a[c] * sum_weight;
    sum_b += b[c] * sum_weight;
  }
  distance *= 4;
  float sum_difference = sum_a - sum_b;
  distance += sum_difference * sum_difference;
  return distance;
}

static int QuantizeColorToImplicitPaletteIndex(
    const std::vector<pixel_type> &color, const int palette_size,
    const int bit_depth, bool high_quality) {
  int index = 0;
  int quant = (1 << bit_depth) - 1;
  // When bit_depth == 1, half would be equal quant without this fuse.
  int half = (bit_depth > 1) ? (1 << (bit_depth - 1)) : 0;
  if (high_quality) {
    int multiplier = 1;
    for (int value : color) {
      int quantized = ((kLargeCube - 1) * value + half) / quant;
      JXL_DASSERT((quantized % kLargeCube) == quantized);
      index += quantized * multiplier;
      multiplier *= kLargeCube;
    }
    return index + palette_size + kLargeCubeOffset;
  } else {
    int multiplier = 1;
    for (int value : color) {
      value -= 1 << (std::max(0, bit_depth - 3));
      value = std::max(0, value);
      int quantized = ((kLargeCube - 1) * value + half) / quant;
      JXL_DASSERT((quantized % kLargeCube) == quantized);
      if (quantized > kSmallCube - 1) {
        quantized = kSmallCube - 1;
      }
      index += quantized * multiplier;
      multiplier *= kSmallCube;
    }
    return index + palette_size;
  }
}

struct DSU {
  std::vector<uint32_t> parent;
  explicit DSU(size_t n) : parent(n) {
    for (size_t i = 0; i < n; ++i) parent[i] = i;
  }
  uint32_t Find(uint32_t x) {
    if (parent[x] != x) parent[x] = Find(parent[x]);
    return parent[x];
  }
  bool Union(uint32_t x, uint32_t y) {
    uint32_t rx = Find(x);
    uint32_t ry = Find(y);
    if (rx == ry) return false;
    parent[rx] = ry;
    return true;
  }
};

struct Edge {
  uint32_t u;
  uint32_t v;
  uint32_t weight;
  uint64_t dist_sq;
};

static uint64_t ColorDistSq(const std::vector<pixel_type>& a,
                            const std::vector<pixel_type>& b) {
  uint64_t dist = 0;
  for (size_t c = 0; c < a.size(); ++c) {
    int64_t diff = static_cast<int64_t>(a[c]) - static_cast<int64_t>(b[c]);
    dist += diff * diff;
  }
  return dist;
}

static void BuildCooccurrenceEdges(
    const Image& input, uint32_t begin_c, uint32_t nb,
    const std::vector<std::vector<pixel_type>>& candidate_palette,
    std::vector<Edge>& edges) {
  size_t K = candidate_palette.size();
  if (K <= 2) return;
  size_t w = input.channel[begin_c].w;
  size_t h = input.channel[begin_c].h;

  struct ColorLookup {
    const std::vector<pixel_type>* color;
    uint32_t id;
  };
  std::vector<ColorLookup> sorted_palette(K);
  for (size_t i = 0; i < K; ++i) {
    sorted_palette[i] = {&candidate_palette[i], static_cast<uint32_t>(i)};
  }
  std::sort(sorted_palette.begin(), sorted_palette.end(),
            [](const ColorLookup& a, const ColorLookup& b) {
              return *a.color < *b.color;
            });

  auto find_color = [&](const std::vector<pixel_type>& c) -> uint32_t {
    ColorLookup dummy{&c, 0};
    auto it = std::lower_bound(
        sorted_palette.begin(), sorted_palette.end(), dummy,
        [](const ColorLookup& a, const ColorLookup& b) {
          return *a.color < *b.color;
        });
    if (it != sorted_palette.end() && *it->color == c) {
      return it->id;
    }
    return static_cast<uint32_t>(-1);
  };

  constexpr uint32_t kInvalidId = static_cast<uint32_t>(-1);
  std::vector<uint32_t> cooccur_dense;
  std::vector<std::map<uint32_t, uint32_t>> cooccur_sparse;
  bool use_dense = (K <= 1024);
  if (use_dense) {
    cooccur_dense.assign(K * K, 0);
  } else {
    cooccur_sparse.resize(K);
  }

  auto add_cooccur = [&](uint32_t u, uint32_t v) {
    if (use_dense) {
      cooccur_dense[u * K + v]++;
      cooccur_dense[v * K + u]++;
    } else {
      if (u > v) std::swap(u, v);
      cooccur_sparse[u][v]++;
    }
  };

  std::vector<uint32_t> prev_row(w, kInvalidId);
  std::vector<uint32_t> curr_row(w, kInvalidId);
  std::vector<pixel_type> pixel_color(nb);

  for (size_t y = 0; y < h; ++y) {
    for (size_t x = 0; x < w; ++x) {
      for (size_t c = 0; c < nb; ++c) {
        pixel_color[c] = input.channel[begin_c + c].Row(y)[x];
      }
      curr_row[x] = find_color(pixel_color);
    }
    for (size_t x = 0; x < w; ++x) {
      uint32_t u = curr_row[x];
      if (u == kInvalidId) continue;
      if (x + 1 < w) {
        uint32_t v = curr_row[x + 1];
        if (v != kInvalidId && u != v) {
          add_cooccur(u, v);
        }
      }
      if (y > 0) {
        uint32_t v = prev_row[x];
        if (v != kInvalidId && u != v) {
          add_cooccur(u, v);
        }
      }
    }
    prev_row.swap(curr_row);
  }

  if (use_dense) {
    for (uint32_t u = 0; u < K; ++u) {
      for (uint32_t v = u + 1; v < K; ++v) {
        uint32_t weight = cooccur_dense[u * K + v];
        if (weight > 0) {
          edges.push_back({u, v, weight,
                           ColorDistSq(candidate_palette[u],
                                       candidate_palette[v])});
        }
      }
    }
  } else {
    for (uint32_t u = 0; u < K; ++u) {
      for (const auto& kv : cooccur_sparse[u]) {
        uint32_t v = kv.first;
        uint32_t weight = kv.second;
        if (weight > 0) {
          edges.push_back({u, v, weight,
                           ColorDistSq(candidate_palette[u],
                                       candidate_palette[v])});
        }
      }
    }
  }

  std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
    if (a.weight != b.weight) return a.weight > b.weight;
    if (a.dist_sq != b.dist_sq) return a.dist_sq < b.dist_sq;
    if (a.u != b.u) return a.u < b.u;
    return a.v < b.v;
  });
}

static void OrientPaletteByFrequency(
    const std::vector<std::vector<pixel_type>>& candidate_palette,
    const std::map<std::vector<pixel_type>, size_t>& color_freq_map,
    std::vector<uint32_t>& order) {
  size_t K = order.size();
  std::vector<size_t> freq(K, 0);
  for (size_t i = 0; i < K; ++i) {
    auto it = color_freq_map.find(candidate_palette[i]);
    if (it != color_freq_map.end()) freq[i] = it->second;
  }
  size_t forward_cost = 0;
  size_t backward_cost = 0;
  for (size_t i = 0; i < K; ++i) {
    forward_cost += i * freq[order[i]];
    backward_cost += (K - 1 - i) * freq[order[i]];
  }
  if (backward_cost < forward_cost) {
    std::reverse(order.begin(), order.end());
  }
}

void OrderPaletteGreedy(
    const Image& input, uint32_t begin_c, uint32_t nb,
    std::vector<std::vector<pixel_type>>& candidate_palette,
    const std::map<std::vector<pixel_type>, size_t>& color_freq_map) {
  size_t K = candidate_palette.size();
  if (K <= 2) return;

  std::vector<Edge> edges;
  BuildCooccurrenceEdges(input, begin_c, nb, candidate_palette, edges);

  DSU dsu(K);
  std::vector<uint8_t> degree(K, 0);
  std::vector<std::vector<uint32_t>> adj(K);
  std::vector<std::pair<uint32_t, uint32_t>> comp_ends(K);
  for (size_t i = 0; i < K; ++i) {
    comp_ends[i] = {static_cast<uint32_t>(i), static_cast<uint32_t>(i)};
  }

  for (const auto& e : edges) {
    if (degree[e.u] < 2 && degree[e.v] < 2) {
      uint32_t root_u = dsu.Find(e.u);
      uint32_t root_v = dsu.Find(e.v);
      if (root_u != root_v) {
        degree[e.u]++;
        degree[e.v]++;
        adj[e.u].push_back(e.v);
        adj[e.v].push_back(e.u);
        uint32_t other_u = (comp_ends[root_u].first == e.u)
                               ? comp_ends[root_u].second
                               : comp_ends[root_u].first;
        uint32_t other_v = (comp_ends[root_v].first == e.v)
                               ? comp_ends[root_v].second
                               : comp_ends[root_v].first;
        dsu.Union(root_u, root_v);
        uint32_t new_root = dsu.Find(root_u);
        comp_ends[new_root] = {other_u, other_v};
      }
    }
  }

  // Connect remaining components using minimum color distance between endpoints
  std::vector<uint32_t> roots;
  for (size_t i = 0; i < K; ++i) {
    if (dsu.Find(i) == i) {
      roots.push_back(i);
    }
  }

  while (roots.size() > 1) {
    size_t best_i = 0;
    size_t best_j = 1;
    uint32_t best_ea = 0;
    uint32_t best_eb = 0;
    uint64_t best_dist = std::numeric_limits<uint64_t>::max();

    for (size_t i = 0; i < roots.size(); ++i) {
      uint32_t ra = roots[i];
      uint32_t ends_a[2] = {comp_ends[ra].first, comp_ends[ra].second};
      for (size_t j = i + 1; j < roots.size(); ++j) {
        uint32_t rb = roots[j];
        uint32_t ends_b[2] = {comp_ends[rb].first, comp_ends[rb].second};
        for (uint32_t ea : ends_a) {
          for (uint32_t eb : ends_b) {
            uint64_t d = ColorDistSq(candidate_palette[ea], candidate_palette[eb]);
            if (d < best_dist) {
              best_dist = d;
              best_i = i;
              best_j = j;
              best_ea = ea;
              best_eb = eb;
            }
          }
        }
      }
    }

    uint32_t ra = roots[best_i];
    uint32_t rb = roots[best_j];
    degree[best_ea]++;
    degree[best_eb]++;
    adj[best_ea].push_back(best_eb);
    adj[best_eb].push_back(best_ea);

    uint32_t other_a = (comp_ends[ra].first == best_ea) ? comp_ends[ra].second
                                                        : comp_ends[ra].first;
    uint32_t other_b = (comp_ends[rb].first == best_eb) ? comp_ends[rb].second
                                                        : comp_ends[rb].first;
    dsu.Union(ra, rb);
    uint32_t new_root = dsu.Find(ra);
    comp_ends[new_root] = {other_a, other_b};

    roots[best_i] = new_root;
    roots.erase(roots.begin() + best_j);
  }

  // Extract path from endpoint to endpoint
  uint32_t final_root = dsu.Find(0);
  uint32_t start_node = comp_ends[final_root].first;
  uint32_t end_node = comp_ends[final_root].second;

  std::vector<uint32_t> order;
  order.reserve(K);
  std::vector<bool> visited(K, false);
  uint32_t curr = start_node;
  order.push_back(curr);
  visited[curr] = true;

  while (curr != end_node) {
    uint32_t next = static_cast<uint32_t>(-1);
    for (uint32_t v : adj[curr]) {
      if (!visited[v]) {
        next = v;
        break;
      }
    }
    JXL_DASSERT(next != static_cast<uint32_t>(-1));
    visited[next] = true;
    order.push_back(next);
    curr = next;
  }

  OrientPaletteByFrequency(candidate_palette, color_freq_map, order);

  std::vector<std::vector<pixel_type>> new_palette(K);
  for (size_t i = 0; i < K; ++i) {
    new_palette[i] = std::move(candidate_palette[order[i]]);
  }
  candidate_palette = std::move(new_palette);
}

static void BuildCooccurrenceMatrix(
    const Image& input, uint32_t begin_c, uint32_t nb,
    const std::vector<std::vector<pixel_type>>& candidate_palette,
    std::vector<uint32_t>& cooccur_matrix) {
  size_t K = candidate_palette.size();
  cooccur_matrix.assign(K * K, 0);
  if (K <= 1) return;
  size_t w = input.channel[begin_c].w;
  size_t h = input.channel[begin_c].h;

  struct ColorLookup {
    const std::vector<pixel_type>* color;
    uint32_t id;
  };
  std::vector<ColorLookup> sorted_palette(K);
  for (size_t i = 0; i < K; ++i) {
    sorted_palette[i] = {&candidate_palette[i], static_cast<uint32_t>(i)};
  }
  std::sort(sorted_palette.begin(), sorted_palette.end(),
            [](const ColorLookup& a, const ColorLookup& b) {
              return *a.color < *b.color;
            });

  auto find_color = [&](const std::vector<pixel_type>& c) -> uint32_t {
    ColorLookup dummy{&c, 0};
    auto it = std::lower_bound(
        sorted_palette.begin(), sorted_palette.end(), dummy,
        [](const ColorLookup& a, const ColorLookup& b) {
          return *a.color < *b.color;
        });
    if (it != sorted_palette.end() && *it->color == c) {
      return it->id;
    }
    return static_cast<uint32_t>(-1);
  };

  constexpr uint32_t kInvalidId = static_cast<uint32_t>(-1);
  std::vector<uint32_t> prev_row(w, kInvalidId);
  std::vector<uint32_t> curr_row(w, kInvalidId);
  std::vector<pixel_type> pixel_color(nb);

  for (size_t y = 0; y < h; ++y) {
    for (size_t x = 0; x < w; ++x) {
      for (size_t c = 0; c < nb; ++c) {
        pixel_color[c] = input.channel[begin_c + c].Row(y)[x];
      }
      curr_row[x] = find_color(pixel_color);
    }
    for (size_t x = 0; x < w; ++x) {
      uint32_t u = curr_row[x];
      if (u == kInvalidId) continue;
      if (x + 1 < w) {
        uint32_t v = curr_row[x + 1];
        if (v != kInvalidId && u != v) {
          cooccur_matrix[u * K + v]++;
          cooccur_matrix[v * K + u]++;
        }
      }
      if (y > 0) {
        uint32_t v = prev_row[x];
        if (v != kInvalidId && u != v) {
          cooccur_matrix[u * K + v]++;
          cooccur_matrix[v * K + u]++;
        }
      }
    }
    prev_row.swap(curr_row);
  }
}

static void ModifiedZengOrdering(
    const std::vector<uint32_t>& cooccur, size_t K,
    const std::vector<std::vector<pixel_type>>& candidate_palette,
    std::vector<uint32_t>& order) {
  order.clear();
  if (K <= 2) {
    for (size_t i = 0; i < K; ++i) order.push_back(i);
    return;
  }

  uint32_t c1 = 0;
  uint64_t max_sum = 0;
  for (size_t i = 0; i < K; ++i) {
    uint64_t sum = 0;
    for (size_t j = 0; j < K; ++j) sum += cooccur[i * K + j];
    if (sum > max_sum) {
      max_sum = sum;
      c1 = i;
    }
  }

  uint32_t c2 = (c1 == 0 ? 1 : 0);
  uint32_t max_c = 0;
  for (size_t i = 0; i < K; ++i) {
    if (i == c1) continue;
    uint32_t w = cooccur[c1 * K + i];
    if (w > max_c) {
      max_c = w;
      c2 = i;
    }
  }

  std::deque<uint32_t> chain;
  chain.push_back(c1);
  chain.push_back(c2);

  std::vector<bool> placed(K, false);
  placed[c1] = true;
  placed[c2] = true;

  std::vector<uint64_t> sums(K, 0);
  for (size_t i = 0; i < K; ++i) {
    if (!placed[i]) {
      sums[i] = cooccur[i * K + c1] + cooccur[i * K + c2];
    }
  }

  for (size_t step = 2; step < K; ++step) {
    uint32_t best_u = 0;
    uint64_t best_val = 0;
    bool found = false;
    for (size_t i = 0; i < K; ++i) {
      if (!placed[i]) {
        if (!found || sums[i] > best_val) {
          best_val = sums[i];
          best_u = i;
          found = true;
        }
      }
    }
    JXL_DASSERT(found);

    if (best_val == 0) {
      uint64_t min_dist = std::numeric_limits<uint64_t>::max();
      uint32_t front_c = chain.front();
      uint32_t back_c = chain.back();
      bool attach_front = false;
      for (size_t i = 0; i < K; ++i) {
        if (!placed[i]) {
          uint64_t df =
              ColorDistSq(candidate_palette[i], candidate_palette[front_c]);
          uint64_t db =
              ColorDistSq(candidate_palette[i], candidate_palette[back_c]);
          if (df < min_dist) {
            min_dist = df;
            best_u = i;
            attach_front = true;
          }
          if (db < min_dist) {
            min_dist = db;
            best_u = i;
            attach_front = false;
          }
        }
      }
      placed[best_u] = true;
      if (attach_front) {
        chain.push_front(best_u);
      } else {
        chain.push_back(best_u);
      }
      for (size_t i = 0; i < K; ++i) {
        if (!placed[i]) sums[i] += cooccur[i * K + best_u];
      }
      continue;
    }

    int64_t delta = 0;
    int32_t m = chain.size();
    for (int32_t j = 0; j < m; ++j) {
      uint32_t lj = chain[j];
      delta += (int64_t)(m - 1 - 2 * j) * (int64_t)cooccur[best_u * K + lj];
    }

    placed[best_u] = true;
    if (delta > 0) {
      chain.push_front(best_u);
    } else {
      chain.push_back(best_u);
    }

    for (size_t i = 0; i < K; ++i) {
      if (!placed[i]) {
        sums[i] += cooccur[i * K + best_u];
      }
    }
  }

  order.assign(chain.begin(), chain.end());
}

static void MinLACutProfile(const std::vector<uint32_t>& cooccur, size_t K,
                            const std::vector<uint32_t>& order,
                            const std::vector<uint64_t>& row_sum,
                            std::vector<int64_t>& cut) {
  cut.assign(K + 1, 0);
  int64_t running = 0;
  cut[0] = 0;
  for (size_t k = 0; k < K; ++k) {
    uint32_t c = order[k];
    const uint32_t* row = &cooccur[c * K];
    int64_t to_before = 0;
    for (size_t m = 0; m < k; ++m) to_before += row[order[m]];
    int64_t to_after = static_cast<int64_t>(row_sum[c]) - to_before;
    running += to_after - to_before;
    cut[k + 1] = running;
  }
}

static uint32_t MinLABestSlot(const std::vector<uint32_t>& cooccur, size_t K,
                              uint32_t at, const std::vector<int64_t>& cut,
                              const std::vector<uint32_t>& order,
                              const std::vector<uint64_t>& row_sum) {
  uint32_t color = order[at];
  int64_t total_w = row_sum[color];
  if (total_w == 0) return at;

  const uint32_t* row = &cooccur[color * K];
  const uint32_t* order_ptr = order.data();
  const int64_t* cut_ptr = cut.data();
  uint32_t m = K - 1;
  int64_t left_w = 0;
  int64_t left_wr = 0;
  int64_t pre_v = 0;
  uint32_t best_j = at;
  int64_t best = std::numeric_limits<int64_t>::max();
  int64_t at_cost = 0;

  for (uint32_t j = 0; j < K; ++j) {
    uint32_t r = (j == m) ? color : (j < at) ? order_ptr[j] : order_ptr[j + 1];
    int64_t own = (int64_t)j * (2 * left_w - total_w) - left_w - 2 * left_wr;
    int64_t straddle =
        (j <= at) ? cut_ptr[j] - pre_v
                  : cut_ptr[j + 1] - (total_w - pre_v - row[order_ptr[j]]);
    int64_t cost = own + straddle;
    if (cost < best) {
      best = cost;
      best_j = j;
    }
    if (j == at) at_cost = cost;
    pre_v += row[order_ptr[j]];
    left_w += row[r];
    left_wr += (int64_t)row[r] * j;
  }
  return (at_cost == best) ? at : best_j;
}

static void PaletteMinLARefine(const std::vector<uint32_t>& cooccur, size_t K,
                               std::vector<uint32_t>& order) {
  if (K <= 2) return;
  std::vector<uint64_t> row_sum(K, 0);
  for (size_t c = 0; c < K; ++c) {
    uint64_t sum = 0;
    const uint32_t* row = &cooccur[c * K];
    for (size_t u = 0; u < K; ++u) sum += row[u];
    row_sum[c] = sum;
  }
  std::vector<int64_t> cut(K + 1);
  MinLACutProfile(cooccur, K, order, row_sum, cut);

  constexpr int kMaxSweeps = 20;
  for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
    int moved = 0;
    for (uint32_t at = 0; at < K; ++at) {
      uint32_t best_j = MinLABestSlot(cooccur, K, at, cut, order, row_sum);
      if (best_j != at) {
        uint32_t color = order[at];
        const uint32_t* row = &cooccur[color * K];
        int64_t c_sum = static_cast<int64_t>(row_sum[color]);

        if (best_j > at) {
          int64_t to_before = 0;
          for (size_t m = 0; m < at; ++m) {
            to_before += row[order[m]];
          }
          for (size_t k = at + 1; k <= best_j; ++k) {
            to_before += row[order[k]];
            cut[k] = cut[k + 1] + 2 * to_before - c_sum;
          }
          for (size_t k = at; k < best_j; ++k) order[k] = order[k + 1];
          order[best_j] = color;
        } else {
          int64_t to_before = 0;
          for (size_t m = 0; m < best_j; ++m) {
            to_before += row[order[m]];
          }
          int64_t prev_cut = cut[best_j];
          for (size_t k = best_j + 1; k <= at; ++k) {
            int64_t next_prev = cut[k];
            cut[k] = prev_cut + c_sum - 2 * to_before;
            prev_cut = next_prev;
            to_before += row[order[k - 1]];
          }
          for (size_t k = at; k > best_j; --k) order[k] = order[k - 1];
          order[best_j] = color;
        }
        ++moved;
      }
    }
    JXL_DEBUG_V(8, "MinLA sweep %d: %d moved", sweep, moved);
    if (moved == 0) break;
  }
}

void OrderPaletteMinLA(
    const Image& input, uint32_t begin_c, uint32_t nb,
    std::vector<std::vector<pixel_type>>& candidate_palette,
    const std::map<std::vector<pixel_type>, size_t>& color_freq_map) {
  size_t K = candidate_palette.size();
  if (K <= 2) return;

  std::vector<uint32_t> cooccur;
  BuildCooccurrenceMatrix(input, begin_c, nb, candidate_palette, cooccur);

  std::vector<uint32_t> order;
  ModifiedZengOrdering(cooccur, K, candidate_palette, order);
  PaletteMinLARefine(cooccur, K, order);

  OrientPaletteByFrequency(candidate_palette, color_freq_map, order);

  std::vector<std::vector<pixel_type>> new_palette(K);
  for (size_t i = 0; i < K; ++i) {
    new_palette[i] = std::move(candidate_palette[order[i]]);
  }
  candidate_palette = std::move(new_palette);
}

void OrderPaletteMinLAGradient(
    const Image& input, uint32_t begin_c, uint32_t nb,
    std::vector<std::vector<pixel_type>>& candidate_palette,
    const std::map<std::vector<pixel_type>, size_t>& color_freq_map) {
  // First, initialize with MinLA ordering.
  OrderPaletteMinLA(input, begin_c, nb, candidate_palette, color_freq_map);

  size_t K = candidate_palette.size();
  if (K <= 2) return;

  size_t w = input.channel[begin_c].w;
  size_t h = input.channel[begin_c].h;
  if (w == 0 || h == 0 || w > 65535 || h > 65535) return;
  if (w * h > 4000000) return;

  // Build fast color lookup.
  struct ColorLookup {
    const std::vector<pixel_type>* color;
    uint32_t id;
  };
  std::vector<ColorLookup> sorted_palette(K);
  for (size_t i = 0; i < K; ++i) {
    sorted_palette[i] = {&candidate_palette[i], static_cast<uint32_t>(i)};
  }
  std::sort(sorted_palette.begin(), sorted_palette.end(),
            [](const ColorLookup& a, const ColorLookup& b) {
              return *a.color < *b.color;
            });

  auto find_color = [&](const std::vector<pixel_type>& c) -> uint32_t {
    ColorLookup dummy{&c, 0};
    auto it = std::lower_bound(
        sorted_palette.begin(), sorted_palette.end(), dummy,
        [](const ColorLookup& a, const ColorLookup& b) {
          return *a.color < *b.color;
        });
    if (it != sorted_palette.end() && *it->color == c) {
      return it->id;
    }
    return static_cast<uint32_t>(-1);
  };

  constexpr uint32_t kInvalidId = static_cast<uint32_t>(-1);
  std::vector<uint32_t> grid(w * h, kInvalidId);
  std::vector<pixel_type> pixel_color(nb);

  for (size_t y = 0; y < h; ++y) {
    for (size_t x = 0; x < w; ++x) {
      for (size_t c = 0; c < nb; ++c) {
        pixel_color[c] = input.channel[begin_c + c].Row(y)[x];
      }
      grid[y * w + x] = find_color(pixel_color);
    }
  }

  std::vector<std::vector<uint32_t>> affected(K);
  for (size_t y = 0; y < h; ++y) {
    for (size_t x = 0; x < w; ++x) {
      uint32_t c = grid[y * w + x];
      if (c >= K) continue;
      uint32_t coord =
          (static_cast<uint32_t>(y) << 16) | static_cast<uint32_t>(x);
      affected[c].push_back(coord);
      if (x + 1 < w) {
        affected[c].push_back(coord + 1);
      }
      if (y + 1 < h) {
        affected[c].push_back(coord + (1 << 16));
      }
      if (x + 1 < w && y + 1 < h) {
        affected[c].push_back(coord + (1 << 16) + 1);
      }
    }
  }

  for (size_t c = 0; c < K; ++c) {
    std::sort(affected[c].begin(), affected[c].end());
    affected[c].erase(std::unique(affected[c].begin(), affected[c].end()),
                      affected[c].end());
  }

  std::vector<uint32_t> order(K);
  std::vector<uint32_t> inv_order(K);
  for (size_t i = 0; i < K; ++i) {
    order[i] = i;
    inv_order[i] = i;
  }

  std::vector<int32_t> log_lut(K + 1);
  for (size_t d = 0; d <= K; ++d) {
    log_lut[d] =
        static_cast<int32_t>(std::round(256.0 * std::log2(1.0 + d)));
  }

  auto eval_pixel = [&](uint32_t coord) -> int32_t {
    size_t x = coord & 0xFFFF;
    size_t y = coord >> 16;
    size_t idx = y * w + x;
    uint32_t c_curr = grid[idx];
    int32_t cur = (c_curr < K) ? inv_order[c_curr] : 0;
    int32_t left = 0;
    if (x > 0) {
      uint32_t c_w = grid[idx - 1];
      left = (c_w < K) ? inv_order[c_w] : 0;
    } else if (y > 0) {
      uint32_t c_n = grid[idx - w];
      left = (c_n < K) ? inv_order[c_n] : 0;
    }
    int32_t top =
        (y > 0) ? ((grid[idx - w] < K) ? inv_order[grid[idx - w]] : 0) : left;
    int32_t topleft = (x > 0 && y > 0)
                          ? ((grid[idx - 1 - w] < K)
                                 ? inv_order[grid[idx - 1 - w]]
                                 : 0)
                          : left;
    int32_t guess = ClampedGradient(top, left, topleft);
    int32_t res = cur - guess;
    uint32_t d = static_cast<uint32_t>(std::abs(res));
    return (d <= K)
               ? log_lut[d]
               : static_cast<int32_t>(std::round(256.0 * std::log2(1.0 + d)));
  };

  auto try_swap = [&](uint32_t pos1, uint32_t pos2) -> bool {
    if (pos1 == pos2) return false;
    uint32_t u = order[pos1];
    uint32_t v = order[pos2];
    const auto& a1 = affected[u];
    const auto& a2 = affected[v];

    int64_t old_cost = 0;
    size_t p1 = 0, p2 = 0;
    while (p1 < a1.size() || p2 < a2.size()) {
      uint32_t pt;
      if (p1 < a1.size() && p2 < a2.size()) {
        if (a1[p1] < a2[p2]) {
          pt = a1[p1++];
        } else if (a2[p2] < a1[p1]) {
          pt = a2[p2++];
        } else {
          pt = a1[p1++];
          p2++;
        }
      } else if (p1 < a1.size()) {
        pt = a1[p1++];
      } else {
        pt = a2[p2++];
      }
      old_cost += eval_pixel(pt);
    }

    std::swap(inv_order[u], inv_order[v]);

    int64_t new_cost = 0;
    p1 = 0;
    p2 = 0;
    bool worse = false;
    while (p1 < a1.size() || p2 < a2.size()) {
      uint32_t pt;
      if (p1 < a1.size() && p2 < a2.size()) {
        if (a1[p1] < a2[p2]) {
          pt = a1[p1++];
        } else if (a2[p2] < a1[p1]) {
          pt = a2[p2++];
        } else {
          pt = a1[p1++];
          p2++;
        }
      } else if (p1 < a1.size()) {
        pt = a1[p1++];
      } else {
        pt = a2[p2++];
      }
      new_cost += eval_pixel(pt);
      if (new_cost >= old_cost) {
        worse = true;
        break;
      }
    }

    if (worse) {
      std::swap(inv_order[u], inv_order[v]);
      return false;
    }

    std::swap(order[pos1], order[pos2]);
    return true;
  };

  // Phase 1: multi-pass adjacent transpositions
  constexpr int kMaxAdjacentPasses = 4;
  for (int pass = 0; pass < kMaxAdjacentPasses; ++pass) {
    int moved = 0;
    for (size_t i = 0; i + 1 < K; ++i) {
      if (try_swap(i, i + 1)) {
        moved++;
      }
    }
    if (moved == 0) break;
  }

  // Phase 2: windowed jump swaps to cross local minima
  for (size_t step : {2, 3, 4, 8, 16}) {
    if (step >= K) continue;
    for (size_t i = 0; i + step < K; ++i) {
      try_swap(i, i + step);
    }
  }

  // Phase 3: settle adjacent transpositions
  for (int pass = 0; pass < 2; ++pass) {
    int moved = 0;
    for (size_t i = 0; i + 1 < K; ++i) {
      if (try_swap(i, i + 1)) {
        moved++;
      }
    }
    if (moved == 0) break;
  }

  std::vector<std::vector<pixel_type>> new_palette(K);
  for (size_t i = 0; i < K; ++i) {
    new_palette[i] = std::move(candidate_palette[order[i]]);
  }
  candidate_palette = std::move(new_palette);
}

}  // namespace palette_internal

int RoundInt(int value, int div) {  // symmetric rounding around 0
  if (value < 0) return -RoundInt(-value, div);
  return (value + div / 2) / div;
}

struct PaletteIterationData {
  static constexpr int kMaxDeltas = 128;
  bool final_run = false;
  std::vector<pixel_type> deltas[3];
  std::vector<double> delta_distances;
  std::vector<pixel_type> frequent_deltas[3];

  // Populates `frequent_deltas` with items from `deltas` based on frequencies
  // and color distances.
  void FindFrequentColorDeltas(int num_pixels, int bitdepth) {
    using pixel_type_3d = std::array<pixel_type, 3>;
    std::map<pixel_type_3d, double> delta_frequency_map;
    pixel_type bucket_size = 3 << std::max(0, bitdepth - 8);
    // Store frequency weighted by delta distance from quantized value.
    for (size_t i = 0; i < deltas[0].size(); ++i) {
      pixel_type_3d delta = {
          {RoundInt(deltas[0][i], bucket_size),
           RoundInt(deltas[1][i], bucket_size),
           RoundInt(deltas[2][i], bucket_size)}};  // a basic form of clustering
      if (delta[0] == 0 && delta[1] == 0 && delta[2] == 0) continue;
      delta_frequency_map[delta] += sqrt(sqrt(delta_distances[i]));
    }

    const float delta_distance_multiplier = 1.0f / num_pixels;

    // Weigh frequencies by magnitude and normalize.
    for (auto &delta_frequency : delta_frequency_map) {
      std::vector<pixel_type> current_delta = {delta_frequency.first[0],
                                               delta_frequency.first[1],
                                               delta_frequency.first[2]};
      float delta_distance =
          std::sqrt(palette_internal::ColorDistance({0, 0, 0}, current_delta)) +
          1;
      delta_frequency.second *=
          static_cast<double>(delta_distance) * delta_distance_multiplier;
    }

    // Sort by weighted frequency.
    using pixel_type_3d_frequency = std::pair<pixel_type_3d, double>;
    std::vector<pixel_type_3d_frequency> sorted_delta_frequency_map(
        delta_frequency_map.begin(), delta_frequency_map.end());
    std::sort(
        sorted_delta_frequency_map.begin(), sorted_delta_frequency_map.end(),
        [](const pixel_type_3d_frequency &a, const pixel_type_3d_frequency &b) {
          return a.second > b.second;
        });

    // Store the top deltas.
    for (auto &delta_frequency : sorted_delta_frequency_map) {
      if (frequent_deltas[0].size() >= kMaxDeltas) break;
      // Number obtained by optimizing on jyrki31 corpus:
      if (delta_frequency.second < 17) break;
      for (int c = 0; c < 3; ++c) {
        frequent_deltas[c].push_back(delta_frequency.first[c] * bucket_size);
      }
    }
  }
};

Status FwdPaletteIteration(Image &input, uint32_t begin_c, uint32_t end_c,
                           uint32_t &nb_colors, uint32_t &nb_deltas,
                           PaletteOrdering ordering, bool lossy,
                           Predictor &predictor,
                           const weighted::Header &wp_header,
                           PaletteIterationData &palette_iteration_data) {
  JXL_QUIET_RETURN_IF_ERROR(CheckEqualChannels(input, begin_c, end_c));
  JXL_ENSURE(begin_c >= input.nb_meta_channels);
  JxlMemoryManager *memory_manager = input.memory_manager();
  uint32_t nb = end_c - begin_c + 1;

  size_t w = input.channel[begin_c].w;
  size_t h = input.channel[begin_c].h;
  if (input.bitdepth >= 32) return false;
  if (!lossy && nb_colors < 2) return false;

  if (!lossy && nb == 1) {
    // Channel palette special case
    if (nb_colors == 0) return false;
    std::vector<pixel_type> lookup;
    pixel_type minval;
    pixel_type maxval;
    compute_minmax(input.channel[begin_c], &minval, &maxval);
    size_t lookup_table_size =
        static_cast<int64_t>(maxval) - static_cast<int64_t>(minval) + 1;
    if (lookup_table_size > palette_internal::kMaxPaletteLookupTableSize) {
      // a lookup table would use too much memory, instead use a slower approach
      // with std::set
      std::set<pixel_type> chpalette;
      pixel_type idx = 0;
      for (size_t y = 0; y < h; y++) {
        const pixel_type *p = input.channel[begin_c].Row(y);
        for (size_t x = 0; x < w; x++) {
          const bool new_color = chpalette.insert(p[x]).second;
          if (new_color) {
            idx++;
            if (idx > static_cast<int>(nb_colors)) return false;
          }
        }
      }
      JXL_DEBUG_V(6, "Channel %i uses only %i colors.", begin_c, idx);
      JXL_ASSIGN_OR_RETURN(Channel pch,
                           Channel::Create(memory_manager, idx, 1));
      pch.hshift = -1;
      pch.vshift = -1;
      nb_colors = idx;
      idx = 0;
      pixel_type *JXL_RESTRICT p_palette = pch.Row(0);
      for (pixel_type p : chpalette) {
        p_palette[idx++] = p;
      }
      for (size_t y = 0; y < h; y++) {
        pixel_type *p = input.channel[begin_c].Row(y);
        for (size_t x = 0; x < w; x++) {
          for (idx = 0;
               p[x] != p_palette[idx] && idx < static_cast<int>(nb_colors);
               idx++) {
            // no-op
          }
          JXL_DASSERT(idx < static_cast<int>(nb_colors));
          p[x] = idx;
        }
      }
      predictor = Predictor::Zero;
      input.nb_meta_channels++;
      input.channel.insert(input.channel.begin(), std::move(pch));

      return true;
    }
    lookup.resize(lookup_table_size, 0);
    pixel_type idx = 0;
    for (size_t y = 0; y < h; y++) {
      const pixel_type *p = input.channel[begin_c].Row(y);
      for (size_t x = 0; x < w; x++) {
        if (lookup[p[x] - minval] == 0) {
          lookup[p[x] - minval] = 1;
          idx++;
          if (idx > static_cast<int>(nb_colors)) return false;
        }
      }
    }
    JXL_DEBUG_V(6, "Channel %i uses only %i colors.", begin_c, idx);
    JXL_ASSIGN_OR_RETURN(Channel pch, Channel::Create(memory_manager, idx, 1));
    pch.hshift = -1;
    pch.vshift = -1;
    nb_colors = idx;
    idx = 0;
    pixel_type *JXL_RESTRICT p_palette = pch.Row(0);
    for (size_t i = 0; i < lookup_table_size; i++) {
      if (lookup[i]) {
        p_palette[idx] = i + minval;
        lookup[i] = idx;
        idx++;
      }
    }
    for (size_t y = 0; y < h; y++) {
      pixel_type *p = input.channel[begin_c].Row(y);
      for (size_t x = 0; x < w; x++) p[x] = lookup[p[x] - minval];
    }
    predictor = Predictor::Zero;
    input.nb_meta_channels++;
    input.channel.insert(input.channel.begin(), std::move(pch));
    return true;
  }

  Image quantized_input(memory_manager);
  if (lossy) {
    JXL_ASSIGN_OR_RETURN(quantized_input, Image::Create(memory_manager, w, h,
                                                        input.bitdepth, nb));
    for (size_t c = 0; c < nb; c++) {
      JXL_RETURN_IF_ERROR(CopyImageTo(input.channel[begin_c + c].plane,
                                      &quantized_input.channel[c].plane));
    }
  }

  JXL_DEBUG_V(
      7, "Trying to represent channels %i-%i using at most a %i-color palette.",
      begin_c, end_c, nb_colors);
  nb_deltas = 0;
  bool delta_used = false;
  std::set<std::vector<pixel_type>> candidate_palette;
  std::vector<std::vector<pixel_type>> candidate_palette_imageorder;
  std::vector<pixel_type> color(nb);
  std::vector<float> color_with_error(nb);
  std::vector<const pixel_type *> p_in(nb);
  std::map<std::vector<pixel_type>, size_t> inv_palette;

  if (lossy) {
    palette_iteration_data.FindFrequentColorDeltas(w * h, input.bitdepth);
    nb_deltas = palette_iteration_data.frequent_deltas[0].size();

    // Count color frequency for colors that make a cross.
    std::map<std::vector<pixel_type>, size_t> color_freq_map;
    for (size_t y = 1; y + 1 < h; y++) {
      for (uint32_t c = 0; c < nb; c++) {
        p_in[c] = input.channel[begin_c + c].Row(y);
      }
      for (size_t x = 1; x + 1 < w; x++) {
        for (uint32_t c = 0; c < nb; c++) {
          color[c] = p_in[c][x];
        }
        int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        bool makes_cross = true;
        for (int i = 0; i < 4 && makes_cross; ++i) {
          int dx = offsets[i][0];
          int dy = offsets[i][1];
          for (uint32_t c = 0; c < nb && makes_cross; c++) {
            if (input.channel[begin_c + c].Row(y + dy)[x + dx] != color[c]) {
              makes_cross = false;
            }
          }
        }
        if (makes_cross) color_freq_map[color] += 1;
      }
    }
    // Add colors satisfying frequency condition to the palette.
    constexpr float kImageFraction = 0.01f;
    size_t color_frequency_lower_bound = 5 + input.h * input.w * kImageFraction;
    for (const auto &color_freq : color_freq_map) {
      if (color_freq.second > color_frequency_lower_bound) {
        candidate_palette.insert(color_freq.first);
        candidate_palette_imageorder.push_back(color_freq.first);
      }
    }
  }

  std::map<std::vector<pixel_type>, bool> implicit_color;
  std::vector<std::vector<pixel_type>> implicit_colors;
  implicit_colors.reserve(palette_internal::kImplicitPaletteSize);
  for (size_t k = 0; k < palette_internal::kImplicitPaletteSize; k++) {
    for (size_t i = 0; i < nb; i++) {
      color[i] = palette_internal::GetPaletteValue(nullptr, k, i, 0, 0,
                                                   input.bitdepth);
    }
    implicit_color[color] = true;
    implicit_colors.push_back(color);
  }

  std::map<std::vector<pixel_type>, size_t> color_freq_map;
  uint32_t implicit_colors_used = 0;
  for (size_t y = 0; y < h; y++) {
    for (uint32_t c = 0; c < nb; c++) {
      p_in[c] = input.channel[begin_c + c].Row(y);
    }
    for (size_t x = 0; x < w; x++) {
      if (lossy && candidate_palette.size() >= nb_colors) break;
      for (uint32_t c = 0; c < nb; c++) {
        color[c] = p_in[c][x];
      }
      const bool new_color = candidate_palette.insert(color).second;
      if (new_color) {
        if (implicit_color[color]) {
          implicit_colors_used++;
        } else {
          candidate_palette_imageorder.push_back(color);
          if (candidate_palette_imageorder.size() > nb_colors) {
            return false;  // too many colors
          }
        }
      }
      color_freq_map[color] += 1;
    }
  }

  nb_colors = nb_deltas + candidate_palette_imageorder.size();

  // not useful to make a single-color palette
  if (!lossy && nb_colors + implicit_colors_used == 1) return false;
  // TODO(jon): if this happens (e.g. solid white group), special-case it for
  // faster encode

  for (size_t k = 0; k < palette_internal::kImplicitPaletteSize; k++) {
    color = implicit_colors[k];
    // still add the color to the explicit palette if it is frequent enough
    if (color_freq_map[color] > 10) {
      nb_colors++;
      candidate_palette_imageorder.push_back(color);
    }
  }
  for (size_t k = 0; k < palette_internal::kImplicitPaletteSize; k++) {
    color = implicit_colors[k];
    inv_palette[color] = nb_colors + k;
  }

  JXL_DEBUG_V(6, "Channels %i-%i can be represented using a %i-color palette.",
              begin_c, end_c, nb_colors);

  JXL_ASSIGN_OR_RETURN(Channel pch,
                       Channel::Create(memory_manager, nb_colors, nb));
  pch.hshift = -1;
  pch.vshift = -1;
  pixel_type *JXL_RESTRICT p_palette = pch.Row(0);
  ptrdiff_t onerow = pch.plane.PixelsPerRow();
  ptrdiff_t onerow_image = input.channel[begin_c].plane.PixelsPerRow();
  const int bit_depth = std::min(input.bitdepth, 24);

  if (lossy) {
    for (uint32_t i = 0; i < nb_deltas; i++) {
      for (size_t c = 0; c < 3; c++) {
        p_palette[c * onerow + i] =
            palette_iteration_data.frequent_deltas[c][i];
      }
    }
  }
  // Separate the palette in two buckets, first the common colors, then the
  // rare colors.
  // Within each bucket, the colors are sorted on luma (times alpha).
  float freq_threshold = 4;  // arbitrary threshold
  int clr = 0;
  if (ordering == PaletteOrdering::kLuma && nb >= 3) {
    JXL_DEBUG_V(7, "Palette of %i colors, using luma order", nb_colors);
    // sort on luma (multiplied by alpha if available)
    std::sort(candidate_palette_imageorder.begin(),
              candidate_palette_imageorder.end(),
              [&](const std::vector<pixel_type>& ap,
                  const std::vector<pixel_type>& bp) {
                float ay;
                float by;
                ay = (0.299f * ap[0] + 0.587f * ap[1] + 0.114f * ap[2] + 0.1f);
                if (ap.size() > 3) ay *= 1.f + ap[3];
                by = (0.299f * bp[0] + 0.587f * bp[1] + 0.114f * bp[2] + 0.1f);
                if (bp.size() > 3) by *= 1.f + bp[3];
                // put common colors first, transparent dark to opaque bright,
                // then rare colors, bright to dark
                ay = color_freq_map[ap] > freq_threshold ? -ay : ay;
                by = color_freq_map[bp] > freq_threshold ? -by : by;
                return ay < by;
              });
  } else if (ordering == PaletteOrdering::kTSPGreedy && nb >= 2) {
    JXL_DEBUG_V(7, "Palette of %i colors, using TSP greedy order", nb_colors);
    palette_internal::OrderPaletteGreedy(
        input, begin_c, nb, candidate_palette_imageorder, color_freq_map);
  } else if (ordering == PaletteOrdering::kMinLA && nb >= 2) {
    JXL_DEBUG_V(7, "Palette of %i colors, using MinLA order", nb_colors);
    palette_internal::OrderPaletteMinLA(
        input, begin_c, nb, candidate_palette_imageorder, color_freq_map);
  } else if (ordering == PaletteOrdering::kMinLAGradient && nb >= 2) {
    JXL_DEBUG_V(7, "Palette of %i colors, using MinLA Gradient order",
                nb_colors);
    palette_internal::OrderPaletteMinLAGradient(
        input, begin_c, nb, candidate_palette_imageorder, color_freq_map);
  } else {
    JXL_DEBUG_V(7, "Palette of %i colors, using image order", nb_colors);
  }

  for (auto pcol : candidate_palette_imageorder) {
    JXL_DEBUG_V(9, "  Color %i :  ", clr);
    for (size_t i = 0; i < nb; i++) {
      p_palette[nb_deltas + i * onerow + clr] = pcol[i];
      JXL_DEBUG_V(9, "%i ", pcol[i]);
    }
    inv_palette[pcol] = clr;
    clr++;
  }
  std::vector<weighted::State> wp_states;
  for (size_t c = 0; c < nb; c++) {
    wp_states.emplace_back(wp_header, w, h);
  }
  std::vector<pixel_type *> p_quant(nb);
  // Three rows of error for dithering: y to y + 2.
  // Each row has two pixels of padding in the ends, which is
  // beneficial for both precision and encoding speed.
  std::vector<std::vector<float>> error_row[3];
  if (lossy) {
    for (auto &row : error_row) {
      row.resize(nb);
      for (size_t c = 0; c < nb; ++c) {
        row[c].resize(w + 4);
      }
    }
  }
  for (size_t y = 0; y < h; y++) {
    for (size_t c = 0; c < nb; c++) {
      p_in[c] = input.channel[begin_c + c].Row(y);
      if (lossy) p_quant[c] = quantized_input.channel[c].Row(y);
    }
    pixel_type *JXL_RESTRICT p = input.channel[begin_c].Row(y);
    for (size_t x = 0; x < w; x++) {
      int index;
      if (!lossy) {
        for (size_t c = 0; c < nb; c++) color[c] = p_in[c][x];
        index = inv_palette[color];
      } else {
        int best_index = 0;
        bool best_is_delta = false;
        float best_distance = std::numeric_limits<float>::infinity();
        std::vector<pixel_type> best_val(nb, 0);
        std::vector<pixel_type> ideal_residual(nb, 0);
        std::vector<pixel_type> quantized_val(nb);
        std::vector<pixel_type> predictions(nb);
        for (double diffusion_multiplier : {0.55, 0.75}) {
          for (size_t c = 0; c < nb; c++) {
            color_with_error[c] =
                p_in[c][x] + (palette_iteration_data.final_run ? 1 : 0) *
                                 diffusion_multiplier * error_row[0][c][x + 2];
            color[c] = Clamp1(lround(color_with_error[c]), 0l,
                              (1l << input.bitdepth) - 1);
          }

          for (size_t c = 0; c < nb; ++c) {
            predictions[c] = PredictNoTreeWP(w, p_quant[c] + x, onerow_image, x,
                                             y, predictor, &wp_states[c])
                                 .guess;
          }
          const auto TryIndex = [&](const int index) {
            for (size_t c = 0; c < nb; c++) {
              quantized_val[c] = palette_internal::GetPaletteValue(
                  p_palette, index, /*c=*/c,
                  /*palette_size=*/nb_colors,
                  /*onerow=*/onerow, /*bit_depth=*/bit_depth);
              if (index < static_cast<int>(nb_deltas)) {
                quantized_val[c] += predictions[c];
              }
            }
            const float color_distance =
                32.0 / (1LL << std::max(0, 2 * (bit_depth - 8))) *
                palette_internal::ColorDistance(color_with_error,
                                                quantized_val);
            float index_penalty = 0;
            if (index == -1) {
              index_penalty = -124;
            } else if (index < 0) {
              index_penalty = -2 * index;
            } else if (index < static_cast<int>(nb_deltas)) {
              index_penalty = 250;
            } else if (index < static_cast<int>(nb_colors)) {
              index_penalty = 150;
            } else if (index < static_cast<int>(nb_colors) +
                                   palette_internal::kLargeCubeOffset) {
              index_penalty = 70;
            } else {
              index_penalty = 256;
            }
            const float distance = color_distance + index_penalty;
            if (distance < best_distance) {
              best_distance = distance;
              best_index = index;
              best_is_delta = index < static_cast<int>(nb_deltas);
              best_val.swap(quantized_val);
              for (size_t c = 0; c < nb; ++c) {
                ideal_residual[c] = color_with_error[c] - predictions[c];
              }
            }
          };
          for (index = palette_internal::kMinImplicitPaletteIndex;
               index < static_cast<int32_t>(nb_colors); index++) {
            TryIndex(index);
          }
          TryIndex(palette_internal::QuantizeColorToImplicitPaletteIndex(
              color, nb_colors, bit_depth,
              /*high_quality=*/false));
          if (palette_internal::kEncodeToHighQualityImplicitPalette) {
            TryIndex(palette_internal::QuantizeColorToImplicitPaletteIndex(
                color, nb_colors, bit_depth,
                /*high_quality=*/true));
          }
        }
        index = best_index;
        delta_used |= best_is_delta;
        if (!palette_iteration_data.final_run) {
          for (size_t c = 0; c < 3; ++c) {
            palette_iteration_data.deltas[c].push_back(ideal_residual[c]);
          }
          palette_iteration_data.delta_distances.push_back(best_distance);
        }

        for (size_t c = 0; c < nb; ++c) {
          wp_states[c].UpdateErrors(best_val[c], x, y, w);
          p_quant[c][x] = best_val[c];
        }
        float len_error = 0;
        for (size_t c = 0; c < nb; ++c) {
          float local_error = color_with_error[c] - best_val[c];
          len_error += local_error * local_error;
        }
        len_error = std::sqrt(len_error);
        float modulate = 1.0;
        int len_limit = 38 << std::max(0, bit_depth - 8);
        if (len_error > len_limit) {
          modulate *= len_limit / len_error;
        }
        for (size_t c = 0; c < nb; ++c) {
          float total_error = (color_with_error[c] - best_val[c]);

          // If the neighboring pixels have some error in the opposite
          // direction of total_error, cancel some or all of it out before
          // spreading among them.
          constexpr int offsets[12][2] = {{1, 2}, {0, 3}, {0, 4}, {1, 1},
                                          {1, 3}, {2, 2}, {1, 0}, {1, 4},
                                          {2, 1}, {2, 3}, {2, 0}, {2, 4}};
          float total_available = 0;
          for (int i = 0; i < 11; ++i) {
            const int row = offsets[i][0];
            const int col = offsets[i][1];
            if (std::signbit(error_row[row][c][x + col]) !=
                std::signbit(total_error)) {
              total_available += error_row[row][c][x + col];
            }
          }
          float weight =
              std::abs(total_error) / (std::abs(total_available) + 1e-3);
          weight = std::min(weight, 1.0f);
          for (int i = 0; i < 11; ++i) {
            const int row = offsets[i][0];
            const int col = offsets[i][1];
            if (std::signbit(error_row[row][c][x + col]) !=
                std::signbit(total_error)) {
              total_error += weight * error_row[row][c][x + col];
              error_row[row][c][x + col] *= (1 - weight);
            }
          }
          total_error *= modulate;
          const float remaining_error = (1.0f / 14.) * total_error;
          error_row[0][c][x + 3] += 2 * remaining_error;
          error_row[0][c][x + 4] += remaining_error;
          error_row[1][c][x + 0] += remaining_error;
          for (int i = 0; i < 5; ++i) {
            error_row[1][c][x + i] += remaining_error;
            error_row[2][c][x + i] += remaining_error;
          }
        }
      }
      if (palette_iteration_data.final_run) p[x] = index;
    }
    if (lossy) {
      for (size_t c = 0; c < nb; ++c) {
        error_row[0][c].swap(error_row[1][c]);
        error_row[1][c].swap(error_row[2][c]);
        std::fill(error_row[2][c].begin(), error_row[2][c].end(), 0.f);
      }
    }
  }
  if (!delta_used) {
    predictor = Predictor::Zero;
  }
  if (palette_iteration_data.final_run) {
    input.nb_meta_channels++;
    input.channel.erase(input.channel.begin() + begin_c + 1,
                        input.channel.begin() + end_c + 1);
    input.channel.insert(input.channel.begin(), std::move(pch));
  }
  nb_colors -= nb_deltas;
  return true;
}

Status FwdPalette(Image &input, uint32_t begin_c, uint32_t end_c,
                  uint32_t &nb_colors, uint32_t &nb_deltas,
                  PaletteOrdering ordering, bool lossy, Predictor &predictor,
                  const weighted::Header &wp_header) {
  PaletteIterationData palette_iteration_data;
  uint32_t nb_colors_orig = nb_colors;
  uint32_t nb_deltas_orig = nb_deltas;
  // preprocessing pass in case of lossy palette
  if (lossy && input.bitdepth >= 8) {
    JXL_RETURN_IF_ERROR(FwdPaletteIteration(
        input, begin_c, end_c, nb_colors_orig, nb_deltas_orig, ordering, lossy,
        predictor, wp_header, palette_iteration_data));
  }
  palette_iteration_data.final_run = true;
  return FwdPaletteIteration(input, begin_c, end_c, nb_colors, nb_deltas,
                             ordering, lossy, predictor, wp_header,
                             palette_iteration_data);
}

}  // namespace jxl
