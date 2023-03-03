#include "fmjxl.h"

#include <assert.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

namespace {
#if defined(_MSC_VER) && !defined(__clang__)
#define FJXL_INLINE __forceinline
FJXL_INLINE uint32_t FloorLog2(uint32_t v) {
  unsigned long index;
  _BitScanReverse(&index, v);
  return index;
}
FJXL_INLINE uint32_t CtzNonZero(uint64_t v) {
  unsigned long index;
  _BitScanForward(&index, v);
  return index;
}
#else
#define FJXL_INLINE inline __attribute__((always_inline))
FJXL_INLINE uint32_t FloorLog2(uint32_t v) {
  return v ? 31 - __builtin_clz(v) : 0;
}
FJXL_INLINE uint32_t CtzNonZero(uint64_t v) { return __builtin_ctzll(v); }
#endif

FJXL_INLINE uint32_t CeilLog2(uint32_t v) {
  return FloorLog2(v) + ((v & (v - 1)) ? 1 : 0);
}

// Compiles to a memcpy on little-endian systems.
FJXL_INLINE void StoreLE64(uint8_t* tgt, uint64_t data) {
#if (!defined(__BYTE_ORDER__) || (__BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__))
  for (int i = 0; i < 8; i++) {
    tgt[i] = (data >> (i * 8)) & 0xFF;
  }
#else
  memcpy(tgt, &data, 8);
#endif
}

template <typename T>
using DataBuf = std::unique_ptr<T[], void (*)(void*)>;

FJXL_INLINE size_t AddBits(uint32_t count, uint64_t bits, uint8_t* data_buf,
                           size_t& bits_in_buffer, uint64_t& bit_buffer) {
  bit_buffer |= bits << bits_in_buffer;
  bits_in_buffer += count;
  StoreLE64(data_buf, bit_buffer);
  size_t bytes_in_buffer = bits_in_buffer / 8;
  bits_in_buffer -= bytes_in_buffer * 8;
  bit_buffer >>= bytes_in_buffer * 8;
  return bytes_in_buffer;
}

struct BitWriter {
  void Allocate(size_t maximum_bit_size) {
    // Leave some padding.
    data.reset(static_cast<uint8_t*>(malloc(maximum_bit_size / 8 + 64)));
  }

  void Rewind() {
    bytes_written = 0;
    bits_in_buffer = 0;
    buffer = 0;
  }

  void Write(uint32_t count, uint64_t bits) {
    bytes_written += AddBits(count, bits, data.get() + bytes_written,
                             bits_in_buffer, buffer);
  }

  void ZeroPadToByte() {
    if (bits_in_buffer != 0) {
      Write(8 - bits_in_buffer, 0);
    }
  }

  DataBuf<uint8_t> data = {nullptr, free};
  size_t bytes_written = 0;
  size_t bits_in_buffer = 0;
  uint64_t buffer = 0;
};

void AddImageHeader(BitWriter* output, size_t w, size_t h) {
  // Signature
  output->Write(16, 0x0AFF);

  // Size header, hand-crafted.
  // Not small
  output->Write(1, 0);

  auto wsz = [output](size_t size) {
    if (size - 1 < (1 << 9)) {
      output->Write(2, 0b00);
      output->Write(9, size - 1);
    } else if (size - 1 < (1 << 13)) {
      output->Write(2, 0b01);
      output->Write(13, size - 1);
    } else if (size - 1 < (1 << 18)) {
      output->Write(2, 0b10);
      output->Write(18, size - 1);
    } else {
      output->Write(2, 0b11);
      output->Write(30, size - 1);
    }
  };

  wsz(h);

  // No special ratio.
  output->Write(3, 0);

  wsz(w);

  // Hand-crafted ImageMetadata.
  output->Write(1, 0);  // all_default
  output->Write(1, 1);  // extra_fields
  output->Write(3, 0);  // orientation
  output->Write(1, 0);  // no intrinsic size
  output->Write(1, 0);  // no preview
  output->Write(1, 1);  // animation
  // TODO(veluca): allow picking FPS?
  output->Write(2, 0b10);   // 30 tps numerator (sel)
  output->Write(10, 29);    // 30 tps numerator
  output->Write(2, 0b00);   // 1 tps denominator
  output->Write(2, 0b01);   // 1 loop (sel)
  output->Write(3, 0b001);  // 1 loop
  output->Write(1, 0);      // no timecodes
  output->Write(1, 0);      // bit_depth.floating_point_sample
  output->Write(2, 0b01);   // bit_depth.bits_per_sample = 10
  output->Write(1, 1);      // 16-bit-buffer sufficient
  output->Write(2, 0b00);   // No extra channel
  output->Write(1, 0);      // Not XYB
  output->Write(1, 1);      // color_encoding.all_default (sRGB)
  output->Write(1, 1);      // tone_mapping.all_default
  output->Write(2, 0b00);   // No extensions.
  output->Write(1, 1);      // all_default transform data
  // No ICC, no preview. Frame should start at byte boundery.
  output->ZeroPadToByte();
}

struct PrefixCode {
  constexpr static size_t kNumSymbols = 16;
  uint8_t nbits[kNumSymbols] = {};
  uint8_t bits[kNumSymbols] = {};

  static uint16_t BitReverse(size_t nbits, uint16_t bits) {
    constexpr uint16_t kNibbleLookup[16] = {
        0b0000, 0b1000, 0b0100, 0b1100, 0b0010, 0b1010, 0b0110, 0b1110,
        0b0001, 0b1001, 0b0101, 0b1101, 0b0011, 0b1011, 0b0111, 0b1111,
    };
    uint16_t rev16 = (kNibbleLookup[bits & 0xF] << 12) |
                     (kNibbleLookup[(bits >> 4) & 0xF] << 8) |
                     (kNibbleLookup[(bits >> 8) & 0xF] << 4) |
                     (kNibbleLookup[bits >> 12]);
    return rev16 >> (16 - nbits);
  }

  // Create the prefix codes given the code lengths.
  static void ComputeCanonicalCode(const uint8_t* nbits, uint8_t* bits,
                                   size_t size) {
    constexpr size_t kMaxCodeLength = 15;
    uint8_t code_length_counts[kMaxCodeLength + 1] = {};
    for (size_t i = 0; i < size; i++) {
      code_length_counts[nbits[i]]++;
      assert(nbits[i] <= kMaxCodeLength);
      assert(nbits[i] <= 8);
    }

    uint16_t next_code[kMaxCodeLength + 1] = {};

    uint16_t code = 0;
    for (size_t i = 1; i < kMaxCodeLength + 1; i++) {
      code = (code + code_length_counts[i - 1]) << 1;
      next_code[i] = code;
    }

    for (size_t i = 0; i < size; i++) {
      bits[i] = BitReverse(nbits[i], next_code[nbits[i]]++);
    }
  }

  // TODO(veluca): this entire logic can likely be replaced with Katajainen.
  template <typename T>
  static void ComputeCodeLengthsNonZeroImpl(const uint64_t* freqs, size_t n,
                                            size_t precision, T infty,
                                            uint8_t* min_limit,
                                            uint8_t* max_limit,
                                            uint8_t* nbits) {
    std::vector<T> dynp(((1U << precision) + 1) * (n + 1), infty);
    auto d = [&](size_t sym, size_t off) -> T& {
      return dynp[sym * ((1 << precision) + 1) + off];
    };
    d(0, 0) = 0;
    for (size_t sym = 0; sym < n; sym++) {
      for (T bits = min_limit[sym]; bits <= max_limit[sym]; bits++) {
        size_t off_delta = 1U << (precision - bits);
        for (size_t off = 0; off + off_delta <= (1U << precision); off++) {
          d(sym + 1, off + off_delta) =
              std::min(d(sym, off) + static_cast<T>(freqs[sym]) * bits,
                       d(sym + 1, off + off_delta));
        }
      }
    }

    size_t sym = n;
    size_t off = 1U << precision;

    assert(d(sym, off) != infty);

    while (sym-- > 0) {
      assert(off > 0);
      for (size_t bits = min_limit[sym]; bits <= max_limit[sym]; bits++) {
        size_t off_delta = 1U << (precision - bits);
        if (off_delta <= off &&
            d(sym + 1, off) == d(sym, off - off_delta) + freqs[sym] * bits) {
          off -= off_delta;
          nbits[sym] = bits;
          break;
        }
      }
    }
  }

  // Computes nbits[i] for i <= n, subject to min_limit[i] <= nbits[i] <=
  // max_limit[i] and sum 2**-nbits[i] == 1, so to minimize sum(nbits[i] *
  // freqs[i]).
  static void ComputeCodeLengthsNonZero(const uint64_t* freqs, size_t n,
                                        uint8_t* min_limit, uint8_t* max_limit,
                                        uint8_t* nbits) {
    size_t precision = 0;
    size_t shortest_length = 255;
    uint64_t freqsum = 0;
    for (size_t i = 0; i < n; i++) {
      assert(freqs[i] != 0);
      freqsum += freqs[i];
      if (min_limit[i] < 1) min_limit[i] = 1;
      assert(min_limit[i] <= max_limit[i]);
      precision = std::max<size_t>(max_limit[i], precision);
      shortest_length = std::min<size_t>(min_limit[i], shortest_length);
    }
    // If all the minimum limits are greater than 1, shift precision so that we
    // behave as if the shortest was 1.
    precision -= shortest_length - 1;
    uint64_t infty = freqsum * precision;
    if (infty < std::numeric_limits<uint32_t>::max() / 2) {
      ComputeCodeLengthsNonZeroImpl(freqs, n, precision,
                                    static_cast<uint32_t>(infty), min_limit,
                                    max_limit, nbits);
    } else {
      ComputeCodeLengthsNonZeroImpl(freqs, n, precision, infty, min_limit,
                                    max_limit, nbits);
    }
  }

  static void ComputeCodeLengths(const uint64_t* freqs, size_t n,
                                 const uint8_t* min_limit_in,
                                 const uint8_t* max_limit_in, uint8_t* nbits) {
    constexpr size_t kMaxSymbolCount = 18;
    assert(n <= kMaxSymbolCount);
    uint64_t compact_freqs[kMaxSymbolCount];
    uint8_t min_limit[kMaxSymbolCount];
    uint8_t max_limit[kMaxSymbolCount];
    size_t ni = 0;
    for (size_t i = 0; i < n; i++) {
      if (freqs[i]) {
        compact_freqs[ni] = freqs[i];
        min_limit[ni] = min_limit_in[i];
        max_limit[ni] = max_limit_in[i];
        ni++;
      }
    }
    uint8_t num_bits[kMaxSymbolCount] = {};
    ComputeCodeLengthsNonZero(compact_freqs, ni, min_limit, max_limit,
                              num_bits);
    ni = 0;
    for (size_t i = 0; i < n; i++) {
      nbits[i] = 0;
      if (freqs[i]) {
        nbits[i] = num_bits[ni++];
      }
    }
  }

  // Invalid code, used to construct arrays.
  PrefixCode() = default;

  explicit PrefixCode(const uint64_t* counts) {
    size_t num = kNumSymbols;
    while (num > 0 && counts[num - 1] == 0) num--;

    constexpr uint8_t kMinLength[kNumSymbols] = {};
    constexpr uint8_t kMaxLength[kNumSymbols] = {8, 8, 8, 8, 8, 8, 8, 8,
                                                 8, 8, 8, 8, 8, 8, 8, 8};

    ComputeCodeLengths(counts, num, kMinLength, kMaxLength, nbits);
    ComputeCanonicalCode(nbits, bits, num);
  }

  void WriteTo(BitWriter* writer) const {
    uint64_t code_length_counts[18] = {};
    for (size_t i = 0; i < kNumSymbols; i++) {
      code_length_counts[nbits[i]]++;
    }
    uint8_t code_length_nbits[18] = {};
    uint8_t code_length_nbits_min[18] = {};
    uint8_t code_length_nbits_max[18] = {
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    };
    ComputeCodeLengths(code_length_counts, 18, code_length_nbits_min,
                       code_length_nbits_max, code_length_nbits);
    writer->Write(2, 0b00);  // HSKIP = 0, i.e. don't skip code lengths.

    // As per Brotli RFC.
    uint8_t code_length_order[18] = {1, 2, 3, 4,  0,  5,  17, 6,  16,
                                     7, 8, 9, 10, 11, 12, 13, 14, 15};
    uint8_t code_length_length_nbits[] = {2, 4, 3, 2, 2, 4};
    uint8_t code_length_length_bits[] = {0, 7, 3, 2, 1, 15};

    // Encode lengths of code lengths.
    size_t num_code_lengths = 18;
    while (code_length_nbits[code_length_order[num_code_lengths - 1]] == 0) {
      num_code_lengths--;
    }
    for (size_t i = 0; i < num_code_lengths; i++) {
      int symbol = code_length_nbits[code_length_order[i]];
      writer->Write(code_length_length_nbits[symbol],
                    code_length_length_bits[symbol]);
    }

    // Compute the canonical codes for the codes that represent the lengths of
    // the actual codes for data.
    uint8_t code_length_bits[18] = {};
    ComputeCanonicalCode(code_length_nbits, code_length_bits, 18);
    // Encode code lengths.
    for (size_t i = 0; i < kNumSymbols; i++) {
      writer->Write(code_length_nbits[nbits[i]], code_length_bits[nbits[i]]);
    }
  }
};

struct PrefixCodeData {
  PrefixCode dc_codes[2][3];
  PrefixCode nnz_codes[2][3];
  PrefixCode ac_codes[2][3];
};

void StoreDCGlobal(BitWriter* writer, bool is_delta,
                   const PrefixCodeData& prefix_codes) {
  // Default DC quant weights. TODO(veluca): this is probably not a good idea.
  writer->Write(1, 1);

  // Quantizer.
  writer->Write(2, 0b10);   // 4097 +
  writer->Write(12, 2047);  // 2047, total of 6144 (~d1) for global scale
  writer->Write(2, 0b00);   // 16 as global scale

  // Non-default block context map (for a smaller / simpler context map later),
  // which puts all transforms of each channel together.
  writer->Write(1, 0);
  writer->Write(16, 0);  // No DC or QF thresholds
  writer->Write(1, 1);   // Simple ctx map.
  writer->Write(2, 2);   // 2 bits per entry
  for (size_t c = 0; c < 3; c++) {
    for (size_t i = 0; i < 13; i++) {  // # of orders
      writer->Write(2, c);
    }
  }

  // Non-default noop CfL.
  writer->Write(1, 0);
  writer->Write(2, 0b00);  // Default color factor
  writer->Write(16, 0);    // 0 (as f16) base correlation x.
  writer->Write(16, 0);    // 0 (as f16) base correlation b.
  writer->Write(8, 128);   // 0 ytox_dc
  writer->Write(8, 128);   // 0 ytob_dc

  // Modular tree
  // TODO(veluca): remove data for channel 4.
  writer->Write(1, 1);         // use global tree
  writer->Write(1, 0);         // no lz77 for context map
  writer->Write(1, 1);         // simple code for the tree's context map
  writer->Write(2, 0);         // all contexts clustered together
  writer->Write(1, 1);         // use prefix code for tree
  writer->Write(4, 0);         // 000 hybrid uint
  writer->Write(6, 0b100011);  // Alphabet size is 4 (var16)
  writer->Write(2, 1);         // simple prefix code
  writer->Write(2, 3);         // with 4 symbols
  writer->Write(2, 0);
  writer->Write(2, 1);
  writer->Write(2, 2);
  writer->Write(2, 3);
  writer->Write(1, 0);  // First tree encoding option
  // Huffman table + extra bits for the tree.
  uint8_t symbol_bits[6] = {0b00, 0b10, 0b001, 0b101, 0b0011, 0b0111};
  uint8_t symbol_nbits[6] = {2, 2, 3, 3, 4, 4};
  // Write a tree with a leaf per channel, and gradient predictor for every
  // leaf.
  for (auto v : {1, 2, 1, 4, 1, 0, 0, 5, 0, 0, 0, 0, 5,
                 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0}) {
    writer->Write(symbol_nbits[v], symbol_bits[v]);
  }

  writer->Write(1, 0);  // no lz77 for the DC bitstream

  writer->Write(1, 1);  // simple code for the context map
  writer->Write(2, 2);  // 3 bits per entry
  writer->Write(2, 2);  // channel 3 (TODO(veluca): remove)
  writer->Write(2, 2);  // channel 2
  writer->Write(2, 1);  // channel 1
  writer->Write(2, 0);  // channel 0

  writer->Write(1, 1);  // use prefix codes
  for (size_t i = 0; i < 3; i++) {
    writer->Write(4, 0);  // 000 hybrid uint config for symbols
  }

  // Symbol alphabet size
  for (size_t i = 0; i < 3; i++) {
    writer->Write(1, 1);  // > 1
    writer->Write(4, 3);  // <= 16
    writer->Write(3, 7);  // == 16
  }

  // Symbol histogram
  for (size_t i = 0; i < 3; i++) {
    prefix_codes.dc_codes[is_delta ? 1 : 0][i].WriteTo(writer);
  }
}

constexpr uint32_t kStandardCoeffOrder[] = {
    0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
};

void StoreACGlobal(BitWriter* writer, size_t num_groups, bool is_delta,
                   const PrefixCodeData& prefix_codes) {
  // All default quantization tables (TODO(veluca): this is probably not a good
  // idea).
  writer->Write(1, 1);

  size_t num_histo_bits = CeilLog2(num_groups);
  writer->Write(num_histo_bits, 0);  // Only one set of histograms.

  // SIMD-friendly coefficient order. TODO(veluca): see if we can do better
  // while still being SIMD-friendly.
  writer->Write(2, 0b11);  // arbitrary mask selector
  writer->Write(13, 0b0000000000001);

  (void)kStandardCoeffOrder;
  constexpr uint8_t kCoeffOrderEncoding[] = {
      0xa6, 0x03, 0x4c, 0xb4, 0x08, 0x11, 0x3a, 0xc6, 0x4a, 0x6f, 0x40, 0x8c,
      0x35, 0x8c, 0x18, 0x8d, 0x06, 0xda, 0x14, 0x04, 0x00, 0xe8, 0xe4, 0x3e,
      0x73, 0xae, 0xd1, 0x8c, 0x5f, 0x03, 0xdd, 0x71, 0x1f, 0x7e, 0xdc, 0xb0,
      0x50, 0xdd, 0xec, 0x28, 0xbd, 0x24, 0x30, 0x8b, 0x41, 0x7b, 0xc4, 0x85,
      0x82, 0x08, 0xfe, 0xed, 0xdd, 0x5c, 0x7b, 0xa8, 0x2e, 0xc7, 0x29, 0xe8,
      0x31, 0xca, 0xb4, 0x9d, 0xe4, 0x5c, 0xc5, 0xec, 0x91, 0x11, 0x33, 0x52,
      0xb9, 0x1d, 0xfe, 0x1c, 0xfe, 0xc9, 0xbf, 0x80, 0xb8, 0x3b, 0x27, 0x51,
      0xd2, 0x19, 0xe2, 0x03, 0x4a, 0x00};
  constexpr size_t kCoeffOrderEncodingBitLength = 716;
  for (size_t i = 0; i < kCoeffOrderEncodingBitLength; i += 8) {
    size_t nb = std::min(kCoeffOrderEncodingBitLength, i + 8) - i;
    writer->Write(nb, kCoeffOrderEncoding[i / 8] & ((1 << nb) - 1));
  }

  writer->Write(1, 0);  // No lz77 for main bitstream.
  // Context map that maps all non-0 contexts to a single context (per channel)
  // and all the coefficient contexts to a single context (also per channel).
  constexpr uint8_t kContextMapEncoding[] = {
      0x06, 0x48, 0xa1, 0x97, 0x46, 0xdc, 0xd8, 0x2e, 0x9a, 0x52, 0x96,
      0x27, 0xf3, 0xa9, 0xa5, 0xd5, 0x96, 0xd6, 0x3a, 0x5a, 0x0c,
  };
  constexpr size_t kContextMapEncodingBitLength = 165;
  for (size_t i = 0; i < kContextMapEncodingBitLength; i += 8) {
    size_t nb = std::min(kContextMapEncodingBitLength, i + 8) - i;
    writer->Write(nb, kContextMapEncoding[i / 8] & ((1 << nb) - 1));
  }

  writer->Write(1, 1);  // use prefix codes
  for (size_t i = 0; i < 6; i++) {
    writer->Write(4, 0);  // 000 hybrid uint config for symbols
  }

  // Symbol alphabet size
  for (size_t i = 0; i < 6; i++) {
    writer->Write(1, 1);  // > 1
    writer->Write(4, 3);  // <= 16
    writer->Write(3, 7);  // == 16
  }

  // Symbol histograms
  for (size_t i = 0; i < 3; i++) {
    prefix_codes.nnz_codes[is_delta ? 1 : 0][i].WriteTo(writer);
  }
  for (size_t i = 0; i < 3; i++) {
    prefix_codes.ac_codes[is_delta ? 1 : 0][i].WriteTo(writer);
  }
}

template <bool is_delta>
void StoreACGroup(BitWriter* writer, const PrefixCodeData& prefix_codes,
                  size_t w, const uint8_t* y_plane, const uint8_t* uv_plane,
                  size_t x0, size_t xs, size_t y0, size_t ys, uint16_t* dc_y,
                  uint16_t* prev_ydct, uint16_t* dc_cb, uint16_t* prev_cbdct,
                  uint16_t* dc_cr, uint16_t* prev_crdct) {
  const PrefixCode* nnz_codes = &prefix_codes.nnz_codes[is_delta ? 0 : 1][0];
  const PrefixCode* ac_codes = &prefix_codes.ac_codes[is_delta ? 0 : 1][0];
  (void)ac_codes;
  for (size_t iy = 0; iy < ys; iy += 8) {
    for (size_t ix = 0; ix < xs; ix += 8) {
      for (size_t c : {1, 0, 2}) {
        if (c != 1 && (ix % 16 != 0 || iy % 16 != 0)) continue;
        writer->Write(nnz_codes[c].nbits[0], nnz_codes[c].bits[0]);
      }
    }
  }
}

void StoreDCGroup(BitWriter* writer, bool is_delta,
                  const PrefixCodeData& prefix_codes, size_t w, size_t x0,
                  size_t xs, size_t y0, size_t ys, const uint16_t* dc_y,
                  const uint16_t* dc_cb, const uint16_t* dc_cr) {
  // No extra DC precision.
  writer->Write(2, 0);

  // Group header for DC modular image.
  writer->Write(1, 1);     // Global tree
  writer->Write(1, 1);     // All default wp
  writer->Write(2, 0b00);  // 0 transforms

  // 1 DC sample per 8 image samples.
  w = w / 8;

  const PrefixCode* codes = &prefix_codes.dc_codes[is_delta ? 0 : 1][0];

  for (size_t c = 0; c < 3; c++) {
    const uint16_t* dc = c == 0 ? dc_y : c == 1 ? dc_cb : dc_cr;
    (void)dc;
    size_t cw = c == 0 ? w : w / 2;
    size_t ch = c == 0 ? w : w / 2;
    size_t cx0 = c == 0 ? x0 : x0 / 2;
    size_t cy0 = c == 0 ? y0 : y0 / 2;
    size_t cxs = c == 0 ? xs : xs / 2;
    size_t cys = c == 0 ? ys : ys / 2;
    (void)cx0;
    (void)cy0;
    (void)cw;
    (void)ch;
    for (size_t y = 0; y < cys; y++) {
      for (size_t x = 0; x < cxs; x++) {
        writer->Write(codes[c].nbits[0], codes[c].bits[0]);
      }
    }
  }

  // AC metadata.
  size_t metadata_count = xs * ys;
  writer->Write(CeilLog2(metadata_count), metadata_count - 1);
  writer->Write(1, 0);     // Custom tree
  writer->Write(1, 1);     // All default wp
  writer->Write(2, 0b00);  // 0 transforms

  writer->Write(1, 0);   // no lz77 for the tree.
  writer->Write(1, 1);   // simple code for the tree's context map
  writer->Write(2, 0);   // all contexts clustered together
  writer->Write(1, 1);   // use prefix code for tree
  writer->Write(4, 15);  // don't do hybriduint for tree - 1 symbol anyway

  writer->Write(1, 0b0);  // Alphabet size is 1: we need 0 only.
  // tree repr is empty.

  writer->Write(1, 0);  // no lz77 for the main data.
  // single ctx, so no ctx map for main data.

  writer->Write(1, 1);    // use prefix code
  writer->Write(4, 15);   // don't do hybriduint
  writer->Write(1, 0b0);  // Alphabet size is 1: we need 0 only.

  // We don't actually need to encode anything for the AC metadata, as
  // everything is 0.
}

}  // namespace

struct FastMJXLEncoder {
  FastMJXLEncoder(size_t w, size_t h) : w(w), h(h) {
    assert(w > 256 || h > 256);
    assert(w % 16 == 0);
    assert(h % 16 == 0);
    prev_ydct.reset(static_cast<uint16_t*>(malloc(w * h * sizeof(uint16_t))));
    prev_cbdct.reset(
        static_cast<uint16_t*>(malloc(w * h * sizeof(uint16_t) / 4)));
    prev_crdct.reset(
        static_cast<uint16_t*>(malloc(w * h * sizeof(uint16_t) / 4)));

    dc_y.reset(static_cast<uint16_t*>(malloc(w * h * sizeof(uint16_t) / 64)));
    dc_cb.reset(static_cast<uint16_t*>(malloc(w * h * sizeof(uint16_t) / 256)));
    dc_cr.reset(static_cast<uint16_t*>(malloc(w * h * sizeof(uint16_t) / 256)));

    num_groups_x = (w + 255) / 256;
    num_groups_y = (h + 255) / 256;
    num_dc_groups_x = (w + 2047) / 2048;
    num_dc_groups_y = (h + 2047) / 2048;

    size_t num_groups =
        2 + num_dc_groups_x * num_dc_groups_y + num_groups_x * num_groups_y;

    group_data = std::vector<BitWriter>(num_groups);
    // 32 bits per pixel should be more than enough.
    // TODO(veluca): figure out a better bound.
    for (size_t i = 0; i < group_data.size(); i++) {
      group_data[i].Allocate(256 * 256 * 32 + 1024);
    }
    encoded.Allocate(w * h * 32 + 1024);

    // TODO(veluca): more sensible prefix codes.
    uint64_t counts[16] = {3843, 1400, 1270, 1214, 1014, 727, 481, 300,
                           159,  51,   5,    1,    1,    1,   1,   1};
    for (size_t j = 0; j < 2; j++) {
      for (size_t i = 0; i < 3; i++) {
        prefix_codes.ac_codes[j][i] = PrefixCode(counts);
        prefix_codes.nnz_codes[j][i] = PrefixCode(counts);
        prefix_codes.dc_codes[j][i] = PrefixCode(counts);
      }
    }
  }

  void AddYCbCrP010Frame(const uint8_t* y_plane, const uint8_t* uv_plane,
                         bool is_last) {
    for (size_t i = 0; i < group_data.size(); i++) {
      group_data[i].Rewind();
    }
    encoded.Rewind();

    if (frame_count == 0) {
      AddImageHeader(&encoded, w, h);
    }

    bool is_delta = frame_count % 8 != 0;

    StoreDCGlobal(group_data.data(), is_delta, prefix_codes);
    size_t acg_off = 2 + num_dc_groups_x * num_dc_groups_y;
    StoreACGlobal(group_data.data() + acg_off - 1, num_groups_x * num_groups_y,
                  is_delta, prefix_codes);

    // TODO(veluca): parallelize both of those loops.
    for (size_t i = 0; i < num_groups_x * num_groups_y; i++) {
      size_t ix = i % num_groups_x;
      size_t iy = i / num_groups_x;
      size_t x0 = ix * 256;
      size_t y0 = iy * 256;
      size_t xs = std::min(w, x0 + 256) - x0;
      size_t ys = std::min(h, y0 + 256) - y0;
      if (is_delta) {
        StoreACGroup<true>(group_data.data() + acg_off + i, prefix_codes, w,
                           y_plane, uv_plane, x0, xs, y0, ys, dc_y.get(),
                           prev_ydct.get(), dc_cb.get(), prev_cbdct.get(),
                           dc_cr.get(), prev_crdct.get());
      } else {
        StoreACGroup<false>(group_data.data() + acg_off + i, prefix_codes, w,
                            y_plane, uv_plane, x0, xs, y0, ys, dc_y.get(),
                            prev_ydct.get(), dc_cb.get(), prev_cbdct.get(),
                            dc_cr.get(), prev_crdct.get());
      }
    }

    for (size_t i = 0; i < num_dc_groups_x * num_dc_groups_y; i++) {
      size_t ix = i % num_dc_groups_x;
      size_t iy = i / num_dc_groups_x;
      size_t x0 = ix * 256;
      size_t y0 = iy * 256;
      size_t xs = std::min(w / 8, x0 + 256) - x0;
      size_t ys = std::min(h / 8, y0 + 256) - y0;
      StoreDCGroup(group_data.data() + 1 + i, is_delta, prefix_codes, w, x0, xs,
                   y0, ys, dc_y.get(), dc_cb.get(), dc_cr.get());
    }

    for (size_t i = 0; i < group_data.size(); i++) {
      group_data[i].ZeroPadToByte();
    }

    // Handcrafted frame header.
    encoded.Write(1, 0);     // all_default
    encoded.Write(2, 0b00);  // regular frame
    encoded.Write(1, 0);     // VarDCT

    // Disable adaptive DC smoothing (flag 128)
    encoded.Write(2, 0b10);
    encoded.Write(8, 111);

    encoded.Write(1, 1);         // YCbCr
    encoded.Write(6, 0b000100);  // 420 downsampling
    encoded.Write(2, 0b00);      // no upsampling
    encoded.Write(2, 0b00);      // exactly one pass
    encoded.Write(1, 0);         // no custom size or origin
    if (is_delta) {
      encoded.Write(2, 0b01);  // kAdd blending mode
      encoded.Write(2, 0b01);  // Add to frame 1.
    } else {
      encoded.Write(2, 0b00);  // kReplace blending mode
    }
    encoded.Write(2, 0b01);     // 1 tick frame
    encoded.Write(1, is_last);  // is_last
    if (!is_last) {
      encoded.Write(2, 0b01);  // Save as frame 1.
      if (!is_delta) {
        encoded.Write(1, 0);  // If this is not a delta frame, save it *after*
                              // color transforms.
      }
    }
    encoded.Write(2, 0b00);  // a frame has no name
    encoded.Write(1, 0);     // loop filter is not all_default
    encoded.Write(1, 0);     // no gaborish
    encoded.Write(2, 0);     // 0 EPF iters
    encoded.Write(2, 0b00);  // No LF extensions
    encoded.Write(2, 0b00);  // No FH extensions

    encoded.Write(1, 0);      // No TOC permutation
    encoded.ZeroPadToByte();  // TOC is byte-aligned.

    for (size_t i = 0; i < group_data.size(); i++) {
      assert(group_data[i].bits_in_buffer == 0);
      size_t sz = group_data[i].bytes_written;
      if (sz < (1 << 10)) {
        encoded.Write(2, 0b00);
        encoded.Write(10, sz);
      } else if (sz - 1024 < (1 << 14)) {
        encoded.Write(2, 0b01);
        encoded.Write(14, sz - 1024);
      } else if (sz - 17408 < (1 << 22)) {
        encoded.Write(2, 0b10);
        encoded.Write(22, sz - 17408);
      } else {
        encoded.Write(2, 0b11);
        encoded.Write(30, sz - 4211712);
      }
    }
    encoded.ZeroPadToByte();  // Groups are byte-aligned.
    for (size_t i = 0; i < group_data.size(); i++) {
      memcpy(encoded.data.get() + encoded.bytes_written,
             group_data[i].data.get(), group_data[i].bytes_written);
      encoded.bytes_written += group_data[i].bytes_written;
    }

    frame_count++;
  }

  size_t w;
  size_t h;
  BitWriter encoded;
  size_t encoded_size = 0;
  size_t frame_count = 0;
  DataBuf<uint16_t> prev_ydct = {nullptr, free};
  DataBuf<uint16_t> prev_cbdct = {nullptr, free};
  DataBuf<uint16_t> prev_crdct = {nullptr, free};

  DataBuf<uint16_t> dc_y = {nullptr, free};
  DataBuf<uint16_t> dc_cb = {nullptr, free};
  DataBuf<uint16_t> dc_cr = {nullptr, free};

  PrefixCodeData prefix_codes;

  std::vector<BitWriter> group_data;
  size_t num_groups_x;
  size_t num_groups_y;
  size_t num_dc_groups_x;
  size_t num_dc_groups_y;
};

#ifdef __cplusplus
extern "C" {
#endif

// TODO(veluca): thread functions.
struct FastMJXLEncoder* FastMJXLCreateEncoder(size_t width, size_t height) {
  return new FastMJXLEncoder(width, height);
}

// Calling this function will clear the output buffer in the encoder of any of
// its current contents. Note that FastMJXLCreateEncoder *will* put bytes in the
// buffer!
// It is invalid to call this function after a call with `is_last = 1`.
void FastMJXLAddYCbCrP010Frame(const uint8_t* y_plane, const uint8_t* uv_plane,
                               int is_last, struct FastMJXLEncoder* encoder) {
  encoder->AddYCbCrP010Frame(y_plane, uv_plane, is_last);
}

const uint8_t* FastMJXLGetOutputBuffer(const struct FastMJXLEncoder* encoder) {
  assert(encoder->encoded.bits_in_buffer == 0);
  return encoder->encoded.data.get();
}

// Returns the number of ready output bytes.
size_t FastMJXLGetOutputBufferSize(const struct FastMJXLEncoder* encoder) {
  assert(encoder->encoded.bits_in_buffer == 0);
  return encoder->encoded.bytes_written;
}

void FastMJXLDestroyEncoder(struct FastMJXLEncoder* encoder) { delete encoder; }

#ifdef __cplusplus
}
#endif
