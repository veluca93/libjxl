#include "fmjxl.h"

#include <arm_neon.h>
#include <assert.h>

#include <cassert>
#include <cmath>
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

  FJXL_INLINE void WriteMultiple(const uint64_t* nbits, const uint64_t* bits,
                                 size_t n) {
    // Necessary because Write() is only guaranteed to work with <=56 bits.
    // Trying to SIMD-fy this code results in lower speed (and definitely less
    // clarity).
    {
      for (size_t i = 0; i < n; i++) {
        this->buffer |= bits[i] << this->bits_in_buffer;
        memcpy(this->data.get() + this->bytes_written, &this->buffer, 8);
        uint64_t shift = 64 - this->bits_in_buffer;
        this->bits_in_buffer += nbits[i];
        // This `if` seems to be faster than using ternaries.
        if (this->bits_in_buffer >= 64) {
          uint64_t next_buffer = bits[i] >> shift;
          this->buffer = next_buffer;
          this->bits_in_buffer -= 64;
          this->bytes_written += 8;
        }
      }
      memcpy(this->data.get() + this->bytes_written, &this->buffer, 8);
      size_t bytes_in_buffer = this->bits_in_buffer / 8;
      this->bits_in_buffer -= bytes_in_buffer * 8;
      this->buffer >>= bytes_in_buffer * 8;
      this->bytes_written += bytes_in_buffer;
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
                                                 7, 7, 7, 7, 7, 7, 7, 7};

    ComputeCodeLengths(counts, num, kMinLength, kMaxLength, nbits);
    ComputeCanonicalCode(nbits, bits, num);
  }

  void WriteTo(BitWriter* writer) const {
    size_t nsym = kNumSymbols;
    while (nsym > 0 && nbits[nsym - 1] == 0) nsym--;
    uint64_t code_length_counts[18] = {};
    for (size_t i = 0; i < nsym; i++) {
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
    for (size_t i = 0; i < nsym; i++) {
      writer->Write(code_length_nbits[nbits[i]], code_length_bits[nbits[i]]);
    }
  }
};

struct PrefixCodeData {
  // Y, Cb, Cr
  PrefixCode dc_codes[2][3];
  // Cb, Y, Cr
  PrefixCode nnz_codes[2][3];
  PrefixCode ac_codes[2][3];
};

void StoreDCGlobal(BitWriter* writer, bool is_delta,
                   const PrefixCodeData& prefix_codes) {
  // DC quant weights.
  writer->Write(1, 0);
  writer->Write(16, 0x3606);  // 1 as f16
  writer->Write(16, 0x3206);  // 0.5 as f16
  writer->Write(16, 0x3606);  // 1 as f16

  // Quantizer.
  writer->Write(2, 0b10);   // 4097 +
  writer->Write(12, 2047);  // 2047, total of 6144 (~d1) for global scale
  writer->Write(2, 0b00);   // 16 as global scale

  // Resulting DC steps: 0.002604, 0.005208, 0.005208

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
    prefix_codes.dc_codes[is_delta ? 0 : 1][i].WriteTo(writer);
  }
}

void StoreACGlobal(BitWriter* writer, size_t num_groups, bool is_delta,
                   const PrefixCodeData& prefix_codes) {
  // Quant tables.
  writer->Write(1, 0);
  // Non-default DCT8 table.
  writer->Write(3, 6);
  writer->Write(4, 2);  // 3 distance bands.

  // TODO(veluca): these tables likely still need some tweaking.

  // cb
  writer->Write(16, 0x4F86);
  writer->Write(16, 0xBC00);
  writer->Write(16, 0xC500);
  // Y
  writer->Write(16, 0x5700);
  writer->Write(16, 0xB400);
  writer->Write(16, 0xBC00);
  // cr
  writer->Write(16, 0x540D);
  writer->Write(16, 0xBE00);
  writer->Write(16, 0xC200);

  // Default for all the other tables.
  for (size_t i = 0; i < 16; i++) {
    writer->Write(3, 0);
  }

  size_t num_histo_bits = CeilLog2(num_groups);
  writer->Write(num_histo_bits, 0);  // Only one set of histograms.

  // SIMD-friendly coefficient order. TODO(veluca): see if we can do better
  // while still being SIMD-friendly.
  writer->Write(2, 0b11);  // arbitrary mask selector
  writer->Write(13, 0b0000000000001);

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
    prefix_codes.nnz_codes[is_delta ? 0 : 1][i].WriteTo(writer);
  }
  for (size_t i = 0; i < 3; i++) {
    prefix_codes.ac_codes[is_delta ? 0 : 1][i].WriteTo(writer);
  }
}

FJXL_INLINE void idct8(int16x8_t data[8]) {
  constexpr float kMatrix[8][8] = {
      {0.125, 0.17337998066526844, 0.16332037060954707, 0.14698445030241986,
       0.12500000000000003, 0.0982118697983878, 0.06764951251827464,
       0.03448742241036789},
      {0.125, 0.14698445030241986, 0.06764951251827464, -0.03448742241036787,
       -0.125, -0.17337998066526844, -0.1633203706095471, -0.09821186979838777},
      {0.125, 0.0982118697983878, -0.06764951251827463, -0.17337998066526844,
       -0.12500000000000003, 0.03448742241036788, 0.16332037060954704,
       0.14698445030241988},
      {0.125, 0.03448742241036789, -0.16332037060954707, -0.09821186979838777,
       0.12499999999999997, 0.14698445030241988, -0.06764951251827465,
       -0.17337998066526847},
      {0.125, -0.03448742241036787, -0.1633203706095471, 0.09821186979838771,
       0.12500000000000003, -0.14698445030241983, -0.06764951251827477,
       0.17337998066526844},
      {0.125, -0.09821186979838774, -0.06764951251827472, 0.17337998066526844,
       -0.12499999999999986, -0.034487422410367834, 0.16332037060954707,
       -0.1469844503024198},
      {0.125, -0.14698445030241986, 0.06764951251827467, 0.034487422410367966,
       -0.12499999999999994, 0.17337998066526847, -0.16332037060954702,
       0.09821186979838765},
      {0.125, -0.17337998066526844, 0.16332037060954704, -0.14698445030241983,
       0.12499999999999985, -0.09821186979838765, 0.06764951251827458,
       -0.03448742241036793}};
  float idct[8][8] = {};
  for (size_t i = 0; i < 8; i++) {
    for (size_t j = 0; j < 8; j++) {
      for (size_t k = 0; k < 8; k++) {
        idct[j][i] += kMatrix[j][k] * data[k][i];
      }
    }
  }
  for (size_t i = 0; i < 8; i++) {
    for (size_t j = 0; j < 8; j++) {
      data[i][j] = std::round(idct[i][j] * 8);
    }
  }
}

// TODO(veluca): validate ranges.
FJXL_INLINE void dct8(int16x8_t data[8]) {
  // TODO(veluca): optimize.
  int16x8_t v0 = data[0];
  int16x8_t v1 = data[7];
  int16x8_t v2 = vhaddq_s16(v0, v1);
  int16x8_t v3 = data[3];
  int16x8_t v4 = data[4];
  int16x8_t v5 = vhaddq_s16(v3, v4);
  int16x8_t v6 = vhaddq_s16(v2, v5);
  int16x8_t v7 = data[1];
  int16x8_t v8 = data[6];
  int16x8_t v9 = vhaddq_s16(v7, v8);
  int16x8_t v10 = data[2];
  int16x8_t v11 = data[5];
  int16x8_t v12 = vhaddq_s16(v10, v11);
  int16x8_t v13 = vhaddq_s16(v9, v12);
  int16x8_t v14 = vhaddq_s16(v6, v13);
  int16x8_t v15 = vhsubq_s16(v0, v1);
  int16x8_t v16 = vqrdmulhq_n_s16(v15, 16705);
  int16x8_t v17 = vhsubq_s16(v3, v4);
  int16x8_t v18_tmp = vqrdmulhq_n_s16(v17, 18446);
  int16x8_t v18 = vmlaq_n_s16(v18_tmp, v17, 2);
  int16x8_t v19 = vhaddq_s16(v16, v18);
  int16x8_t v20 = vhsubq_s16(v7, v8);
  int16x8_t v21 = vqrdmulhq_n_s16(v20, 19705);
  int16x8_t v22 = vhsubq_s16(v10, v11);
  int16x8_t v23 = vqrdmulhq_n_s16(v22, 29490);
  int16x8_t v24 = vhaddq_s16(v21, v23);
  int16x8_t v25 = vhaddq_s16(v19, v24);
  int16x8_t v26_tmp = vqrdmulhq_n_s16(v25, 13573);
  int16x8_t v26 = vaddq_s16(v26_tmp, v25);
  int16x8_t v27 = vhsubq_s16(v16, v18);
  int16x8_t v28 = vqrdmulhq_n_s16(v27, 17734);
  int16x8_t v29 = vhsubq_s16(v21, v23);
  int16x8_t v30_tmp = vqrdmulhq_n_s16(v29, 10045);
  int16x8_t v30 = vaddq_s16(v30_tmp, v29);
  int16x8_t v31 = vhaddq_s16(v28, v30);
  int16x8_t v32_tmp = vqrdmulhq_n_s16(v31, 13573);
  int16x8_t v32 = vaddq_s16(v32_tmp, v31);
  int16x8_t v33 = vhsubq_s16(v28, v30);
  int16x8_t v34 = vaddq_s16(v32, v33);
  int16x8_t v35 = vaddq_s16(v26, v34);
  int16x8_t v36 = vhsubq_s16(v2, v5);
  int16x8_t v37 = vqrdmulhq_n_s16(v36, 17734);
  int16x8_t v38 = vhsubq_s16(v9, v12);
  int16x8_t v39_tmp = vqrdmulhq_n_s16(v38, 10045);
  int16x8_t v39 = vaddq_s16(v39_tmp, v38);
  int16x8_t v40 = vhaddq_s16(v37, v39);
  int16x8_t v41_tmp = vqrdmulhq_n_s16(v40, 13573);
  int16x8_t v41 = vaddq_s16(v41_tmp, v40);
  int16x8_t v42 = vhsubq_s16(v37, v39);
  int16x8_t v43 = vaddq_s16(v41, v42);
  int16x8_t v44 = vhsubq_s16(v19, v24);
  int16x8_t v45 = vaddq_s16(v34, v44);
  int16x8_t v46 = vhsubq_s16(v6, v13);
  int16x8_t v47 = vaddq_s16(v44, v33);
  data[0] = v14;
  data[1] = v35;
  data[2] = v43;
  data[3] = v45;
  data[4] = v46;
  data[5] = v47;
  data[6] = v42;
  data[7] = v33;
}

FJXL_INLINE void transpose8(int16x8_t data[8]) {
  auto t0 = vtrn1q_s16(data[0], data[1]);
  auto t1 = vtrn2q_s16(data[0], data[1]);
  auto t2 = vtrn1q_s16(data[2], data[3]);
  auto t3 = vtrn2q_s16(data[2], data[3]);
  auto t4 = vtrn1q_s16(data[4], data[5]);
  auto t5 = vtrn2q_s16(data[4], data[5]);
  auto t6 = vtrn1q_s16(data[6], data[7]);
  auto t7 = vtrn2q_s16(data[6], data[7]);
  auto u0 = vtrn1q_s32(vreinterpretq_s32_s16(t0), vreinterpretq_s32_s16(t2));
  auto u1 = vtrn2q_s32(vreinterpretq_s32_s16(t0), vreinterpretq_s32_s16(t2));
  auto u2 = vtrn1q_s32(vreinterpretq_s32_s16(t1), vreinterpretq_s32_s16(t3));
  auto u3 = vtrn2q_s32(vreinterpretq_s32_s16(t1), vreinterpretq_s32_s16(t3));
  auto u4 = vtrn1q_s32(vreinterpretq_s32_s16(t4), vreinterpretq_s32_s16(t6));
  auto u5 = vtrn2q_s32(vreinterpretq_s32_s16(t4), vreinterpretq_s32_s16(t6));
  auto u6 = vtrn1q_s32(vreinterpretq_s32_s16(t5), vreinterpretq_s32_s16(t7));
  auto u7 = vtrn2q_s32(vreinterpretq_s32_s16(t5), vreinterpretq_s32_s16(t7));
  auto v0 = vtrn1q_s64(vreinterpretq_s64_s32(u0), vreinterpretq_s64_s32(u4));
  auto v1 = vtrn2q_s64(vreinterpretq_s64_s32(u0), vreinterpretq_s64_s32(u4));
  auto v2 = vtrn1q_s64(vreinterpretq_s64_s32(u1), vreinterpretq_s64_s32(u5));
  auto v3 = vtrn2q_s64(vreinterpretq_s64_s32(u1), vreinterpretq_s64_s32(u5));
  auto v4 = vtrn1q_s64(vreinterpretq_s64_s32(u2), vreinterpretq_s64_s32(u6));
  auto v5 = vtrn2q_s64(vreinterpretq_s64_s32(u2), vreinterpretq_s64_s32(u6));
  auto v6 = vtrn1q_s64(vreinterpretq_s64_s32(u3), vreinterpretq_s64_s32(u7));
  auto v7 = vtrn2q_s64(vreinterpretq_s64_s32(u3), vreinterpretq_s64_s32(u7));
  data[0] = vreinterpretq_s16_s64(v0);
  data[1] = vreinterpretq_s16_s64(v4);
  data[2] = vreinterpretq_s16_s64(v2);
  data[3] = vreinterpretq_s16_s64(v6);
  data[4] = vreinterpretq_s16_s64(v1);
  data[5] = vreinterpretq_s16_s64(v5);
  data[6] = vreinterpretq_s16_s64(v3);
  data[7] = vreinterpretq_s16_s64(v7);
}

// TODO(veluca): adjust.
constexpr static int16_t kQuantMatrix[3][64] = {
    {
        0x3fc0, 0x139f, 0x110f, 0x0ed4, 0x0ce4, 0x0b15, 0x07b7, 0x055f,  //
        0x139f, 0x1284, 0x1081, 0x0e7f, 0x0cac, 0x0ab1, 0x077d, 0x053c,  //
        0x110f, 0x1081, 0x0f30, 0x0d9f, 0x0c11, 0x09a4, 0x06dc, 0x04db,  //
        0x0ed4, 0x0e7f, 0x0d9f, 0x0c76, 0x0b15, 0x0834, 0x05f9, 0x044d,  //
        0x0ce4, 0x0cac, 0x0c11, 0x0b15, 0x08bd, 0x06ab, 0x04fa, 0x03a8,  //
        0x0b15, 0x0ab1, 0x09a4, 0x0834, 0x06ab, 0x053c, 0x0402, 0x0302,  //
        0x07b7, 0x077d, 0x06dc, 0x05f9, 0x04fa, 0x0402, 0x0323, 0x0268,  //
        0x055f, 0x053c, 0x04db, 0x044d, 0x03a8, 0x0302, 0x0268, 0x01e2,  //
    },
    {
        0x7f80, 0x504c, 0x4cc2, 0x4960, 0x4624, 0x42bb, 0x3a02, 0x326e,  //
        0x504c, 0x4ed0, 0x4bf2, 0x48d7, 0x45c0, 0x41cf, 0x3957, 0x31ee,  //
        0x4cc2, 0x4bf2, 0x49f2, 0x4766, 0x44aa, 0x3f3a, 0x376e, 0x307d,  //
        0x4960, 0x48d7, 0x4766, 0x4560, 0x42bb, 0x3b66, 0x3488, 0x2e43,  //
        0x4624, 0x45c0, 0x44aa, 0x42bb, 0x3cdd, 0x36d3, 0x30f6, 0x2b75,  //
        0x42bb, 0x41cf, 0x3f3a, 0x3b66, 0x36d3, 0x31ee, 0x2d05, 0x284b,  //
        0x3a02, 0x3957, 0x376e, 0x3488, 0x30f6, 0x2d05, 0x28f5, 0x24f5,  //
        0x326e, 0x31ee, 0x307d, 0x2e43, 0x2b75, 0x284b, 0x24f5, 0x219a,  //
    },
    {
        0x3fc0, 0x2865, 0x2191, 0x1be5, 0x172e, 0x132c, 0x0e7d, 0x0af3,  //
        0x2865, 0x256a, 0x2022, 0x1b12, 0x16a9, 0x12a6, 0x0e28, 0x0abc,  //
        0x2191, 0x2022, 0x1ccc, 0x18f0, 0x153e, 0x1136, 0x0d3b, 0x0a20,  //
        0x1be5, 0x1b12, 0x18f0, 0x162a, 0x132c, 0x0f31, 0x0be2, 0x0937,  //
        0x172e, 0x16a9, 0x153e, 0x132c, 0x0ff3, 0x0cf1, 0x0a52, 0x0822,  //
        0x132c, 0x12a6, 0x1136, 0x0f31, 0x0cf1, 0x0abc, 0x08ba, 0x06fe,  //
        0x0e7d, 0x0e28, 0x0d3b, 0x0be2, 0x0a52, 0x08ba, 0x0739, 0x05e1,  //
        0x0af3, 0x0abc, 0x0a20, 0x0937, 0x0822, 0x06fe, 0x05e1, 0x04dc,  //
    }};

constexpr size_t kInputShift = 2;
constexpr size_t kQuantizeShift = 6 - kInputShift;
constexpr size_t kChannelCenterOffset = (1 << (16 - kInputShift)) * 128 / 255;

FJXL_INLINE void scale_inputs(int16x8_t data[8]) {
  for (size_t i = 0; i < 8; i++) {
    int16x8_t v = data[i];
    v = vreinterpretq_s16_u16(
        vrshrq_n_u16(vreinterpretq_u16_s16(v), kInputShift - 1));
    data[i] = vqrdmulhq_n_s16(v, 16399);
  }
}

FJXL_INLINE void quantize(int16x8_t data[8], int c) {
  for (size_t i = 0; i < 8; i++) {
    int16x8_t q = vld1q_s16(&kQuantMatrix[c][i * 8]);
    data[i] = vrshrq_n_s16(vqrdmulhq_s16(q, data[i]), kQuantizeShift);
  }
}

FJXL_INLINE constexpr uint32_t PackSigned(int32_t value) {
  return (static_cast<uint32_t>(value) << 1) ^
         ((static_cast<uint32_t>(~value) >> 31) - 1);
}
FJXL_INLINE void EncodeHybridUint000(uint32_t value, uint32_t* token,
                                     uint32_t* nbits, uint32_t* bits) {
  uint32_t n = FloorLog2(value);
  *token = value ? n + 1 : 0;
  *nbits = value ? n : 0;
  *bits = value ? value - (1 << n) : 0;
}

static constexpr bool kPrintSymbols = false;

void EncodeU32(uint32_t value, const PrefixCode& code, BitWriter* writer,
               const char* tag, size_t c, bool is_delta) {
  uint32_t token, nbits, bits;
  EncodeHybridUint000(value, &token, &nbits, &bits);
  if (kPrintSymbols) {
    static FILE* symbols = fopen("/tmp/symbols.txt", "w");
    fprintf(symbols, "%s %zu %d %u\n", tag, c, is_delta, token);
  }
  writer->Write(code.nbits[token] + nbits,
                code.bits[token] | (bits << code.nbits[token]));
}

FJXL_INLINE void SIMDStoreValues(int16x8_t coeffs, uint16x8_t mask,
                                 const PrefixCode& code, uint64_t* nbits_s,
                                 uint64_t* bits_s) {
  // coeff to token + extra bits
  int16x8_t coeffst2 = vaddq_s16(coeffs, coeffs);
  int16x8_t zero = vdupq_n_s16(0);
  uint16x8_t ucoeffs = vreinterpretq_u16_s16(vbslq_s16(
      vcgeq_s16(coeffs, zero), coeffst2, vsubq_s16(vdupq_n_s16(-1), coeffst2)));

  uint16x8_t token = vsubq_u16(vdupq_n_u16(16), vclzq_u16(ucoeffs));
  int16x8_t nbits = vreinterpretq_s16_u16(vqsubq_u16(token, vdupq_n_u16(1)));
  uint16x8_t bits = vqsubq_u16(ucoeffs, vshlq_u16(vdupq_n_u16(1), nbits));

  // Prefix code lookup.
  uint8x16_t token_indices =
      vreinterpretq_u8_u16(vorrq_u16(token, vdupq_n_u16(0xFF00)));
  uint8x16_t tbl_nbits = vld1q_u8(code.nbits);
  uint8x16_t tbl_bits = vld1q_u8(code.bits);
  int16x8_t tnbits = vreinterpretq_s16_u8(vqtbl1q_u8(tbl_nbits, token_indices));
  uint16x8_t tbits = vreinterpretq_u16_u8(vqtbl1q_u8(tbl_bits, token_indices));

  // merge bits and token bits
  bits = vorrq_u16(tbits, vshlq_u16(bits, tnbits));
  nbits = vaddq_s16(nbits, tnbits);

  // clear out masked out coefficients
  nbits = vandq_s16(nbits, vreinterpretq_s16_u16(mask));
  bits = vandq_u16(bits, mask);

  // Merge 16-bits bits to 32-bits
  int32x4_t nbits_lo16 =
      vandq_s32(vreinterpretq_s32_s16(nbits), vdupq_n_s32(0xFFFF));
  uint32x4_t bits_hi16 =
      vshlq_u32(vshrq_n_u32(vreinterpretq_u32_u16(bits), 16), nbits_lo16);
  uint32x4_t bits_lo16 =
      vandq_u32(vreinterpretq_u32_u16(bits), vdupq_n_u32(0xFFFF));
  int32x4_t nbits32 = vsraq_n_s32(nbits_lo16, vreinterpretq_s32_s16(nbits), 16);
  uint32x4_t bits32 = vorrq_u32(bits_hi16, bits_lo16);

  // 32 -> 64
  uint64x2_t nbits_lo32 =
      vandq_u64(vreinterpretq_u64_s32(nbits32), vdupq_n_u64(0xFFFFFFFF));
  uint64x2_t bits_hi32 =
      vshlq_u64(vshrq_n_u64(vreinterpretq_u64_u32(bits32), 32),
                vreinterpretq_s64_u64(nbits_lo32));
  uint64x2_t bits_lo32 =
      vandq_u64(vreinterpretq_u64_u32(bits32), vdupq_n_u64(0xFFFFFFFF));
  uint64x2_t nbits64 =
      vsraq_n_u64(nbits_lo32, vreinterpretq_u64_s32(nbits32), 32);
  uint64x2_t bits64 = vorrq_u64(bits_hi32, bits_lo32);

  // Final write-out
  vst1q_u64(bits_s, bits64);
  vst1q_u64(nbits_s, nbits64);
}

template <bool is_delta, bool store_for_next>
FJXL_INLINE void ProcessBlock(int16x8_t data[8], int16_t* storage) {
  scale_inputs(data);
  dct8(data);
  transpose8(data);
  dct8(data);
  data[0][0] -= kChannelCenterOffset;
  for (size_t i = 0; i < 8; i++) {
    if (is_delta) {
      int16x8_t prev = vld1q_s16(storage + i * 8);
      if (store_for_next) {
        vst1q_s16(storage + i * 8, data[i]);
      }
      data[i] = vsubq_s16(data[i], prev);
    } else if (store_for_next) {
      vst1q_s16(storage + i * 8, data[i]);
    }
  }
}

template <bool is_delta, bool store_for_next>
FJXL_INLINE void QuantizeAndStore(int16x8_t data[8], int16_t* dc_ptr, size_t c,
                                  const PrefixCode& nnz_code,
                                  const PrefixCode& ac_code,
                                  BitWriter* writer) {
  quantize(data, c);
  *dc_ptr = data[0][0];

  int16x8_t shuffled_data[8];

  constexpr uint16_t kDCMask[8] = {0x0000, 0xFFFF, 0xFFFF, 0xFFFF,
                                   0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF};

  /*
    0,  1,  2,  3,  8,  9,  10, 11, 16, 17, 18, 19, 24, 25, 26, 27,
    4,  5,  6,  7,  12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31,
    32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59,
    36, 37, 38, 39, 44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63,
   */
  shuffled_data[0] = vreinterpretq_s16_s64(vtrn1q_s64(
      vreinterpretq_s64_s16(data[0]), vreinterpretq_s64_s16(data[1])));
  shuffled_data[1] = vreinterpretq_s16_s64(vtrn1q_s64(
      vreinterpretq_s64_s16(data[2]), vreinterpretq_s64_s16(data[3])));
  shuffled_data[2] = vreinterpretq_s16_s64(vtrn2q_s64(
      vreinterpretq_s64_s16(data[0]), vreinterpretq_s64_s16(data[1])));
  shuffled_data[3] = vreinterpretq_s16_s64(vtrn2q_s64(
      vreinterpretq_s64_s16(data[2]), vreinterpretq_s64_s16(data[3])));
  shuffled_data[4] = vreinterpretq_s16_s64(vtrn1q_s64(
      vreinterpretq_s64_s16(data[4]), vreinterpretq_s64_s16(data[5])));
  shuffled_data[5] = vreinterpretq_s16_s64(vtrn1q_s64(
      vreinterpretq_s64_s16(data[6]), vreinterpretq_s64_s16(data[7])));
  shuffled_data[6] = vreinterpretq_s16_s64(vtrn2q_s64(
      vreinterpretq_s64_s16(data[4]), vreinterpretq_s64_s16(data[5])));
  shuffled_data[7] = vreinterpretq_s16_s64(vtrn2q_s64(
      vreinterpretq_s64_s16(data[6]), vreinterpretq_s64_s16(data[7])));

  uint16x8_t dc_mask = vld1q_u16(kDCMask);
  shuffled_data[0] =
      vandq_s16(shuffled_data[0], vreinterpretq_s16_u16(dc_mask));

  uint8_t nnz_vec[8];
  size_t nnz = 0;
  for (size_t i = 0; i < 8; i++) {
    nnz_vec[i] = 8 + vaddvq_s16(vreinterpretq_s16_u16(
                         vceqq_s16(shuffled_data[i], vdupq_n_s16(0))));
    nnz += nnz_vec[i];
  }

  EncodeU32(nnz, nnz_code, writer, "NNZ", c, is_delta);

  constexpr uint16_t kNumGreaterLanes[8] = {7, 6, 5, 4, 3, 2, 1, 0};

  constexpr uint16_t kMask[16] = {
      0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
      0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000};

  if (nnz == 0) return;

  uint64_t nbits_s[16];
  uint64_t bits_s[16];

  size_t to_write = 0;
  for (size_t i = 0; i < 8; i++) {
    nnz -= nnz_vec[i];

    uint16x8_t isz = vceqq_s16(shuffled_data[i], vdupq_n_s16(0));
    uint16x8_t nzloc =
        vbslq_u16(isz, vdupq_n_u16(8), vld1q_u16(kNumGreaterLanes));
    uint16_t trailingzeros = vminvq_u16(nzloc);
    uint16_t maskpos = nnz == 0 ? trailingzeros : 0;

    const uint16_t* maskptr = kMask + maskpos;

    if (kPrintSymbols) {
      const uint16_t* dc_mask = i == 0 ? kDCMask : kMask;
      int16_t buf[8];
      vst1q_s16(buf, shuffled_data[i]);
      for (size_t j = 0; j < 8; j++) {
        int16_t coeff = buf[j];
        if (maskptr[j] & dc_mask[j]) {
          EncodeU32(PackSigned(coeff), ac_code, writer, "AC", c, is_delta);
        }
      }
    } else {
      uint16x8_t mask = vld1q_u16(maskptr);
      if (i == 0) mask = vandq_u16(mask, dc_mask);
      SIMDStoreValues(shuffled_data[i], mask, ac_code, nbits_s + to_write,
                      bits_s + to_write);
      to_write += 2;
    }
    if (nnz == 0) break;
  }
  writer->WriteMultiple(nbits_s, bits_s, to_write);
}

template <bool is_delta, bool store_for_next>
void StoreACGroup(BitWriter* writer, const PrefixCodeData& prefix_codes,
                  size_t w, const uint8_t* y_plane, const uint8_t* uv_plane,
                  size_t x0, size_t xs, size_t y0, size_t ys, int16_t* dc_y,
                  int16_t* prev_ydct, int16_t* dc_cb, int16_t* prev_cbdct,
                  int16_t* dc_cr, int16_t* prev_crdct) {
  const PrefixCode* nnz_codes = &prefix_codes.nnz_codes[is_delta ? 0 : 1][0];
  const PrefixCode* ac_codes = &prefix_codes.ac_codes[is_delta ? 0 : 1][0];

  for (size_t iy = 0; iy < ys; iy += 8) {
    for (size_t ix = 0; ix < xs; ix += 8) {
      size_t bx = (ix + x0) / 8;
      size_t by = (iy + y0) / 8;
      size_t block_idx = by * w / 8 + bx;
      const int16_t* y_ptr =
          reinterpret_cast<const int16_t*>(y_plane) + (y0 + iy) * w + x0 + ix;
      int16x8_t y_data[8];
      for (size_t i = 0; i < 8; i++) {
        y_data[i] = vld1q_s16(y_ptr + w * i);
      }

      int16_t* yblock_ptr = prev_ydct + block_idx * 64;
      ProcessBlock<is_delta, store_for_next>(y_data, yblock_ptr);

      // Adjust Y channel.
      yblock_ptr[0] += kChannelCenterOffset;

      QuantizeAndStore<is_delta, store_for_next>(
          y_data, dc_y + block_idx, 1, nnz_codes[1], ac_codes[1], writer);

      if (ix % 16 == 0 && iy % 16 == 0) {
        size_t cblock_idx = by / 2 * w / 16 + bx / 2;
        const int16_t* chroma_ptr = reinterpret_cast<const int16_t*>(uv_plane) +
                                    ((y0 + iy) / 2) * w + x0 + ix;
        int16x8_t cb_data[8];
        int16x8_t cr_data[8];
        for (size_t i = 0; i < 8; i++) {
          auto v = vld2q_s16(chroma_ptr + w * i);
          cb_data[i] = v.val[0];
          cr_data[i] = v.val[1];
        }
        ProcessBlock<is_delta, store_for_next>(cb_data,
                                               prev_cbdct + cblock_idx * 64);
        QuantizeAndStore<is_delta, store_for_next>(
            cb_data, dc_cb + cblock_idx, 0, nnz_codes[0], ac_codes[0], writer);
        ProcessBlock<is_delta, store_for_next>(cr_data,
                                               prev_crdct + cblock_idx * 64);
        QuantizeAndStore<is_delta, store_for_next>(
            cr_data, dc_cr + cblock_idx, 2, nnz_codes[2], ac_codes[2], writer);
      }
    }
  }
}

template <bool is_delta>
void StoreACGroup(bool store_for_next, BitWriter* writer,
                  const PrefixCodeData& prefix_codes, size_t w,
                  const uint8_t* y_plane, const uint8_t* uv_plane, size_t x0,
                  size_t xs, size_t y0, size_t ys, int16_t* dc_y,
                  int16_t* prev_ydct, int16_t* dc_cb, int16_t* prev_cbdct,
                  int16_t* dc_cr, int16_t* prev_crdct) {
  if (store_for_next) {
    StoreACGroup<is_delta, true>(writer, prefix_codes, w, y_plane, uv_plane, x0,
                                 xs, y0, ys, dc_y, prev_ydct, dc_cb, prev_cbdct,
                                 dc_cr, prev_crdct);
  } else {
    StoreACGroup<is_delta, false>(writer, prefix_codes, w, y_plane, uv_plane,
                                  x0, xs, y0, ys, dc_y, prev_ydct, dc_cb,
                                  prev_cbdct, dc_cr, prev_crdct);
  }
}

void StoreDCGroup(BitWriter* writer, bool is_delta,
                  const PrefixCodeData& prefix_codes, size_t w, size_t x0,
                  size_t xs, size_t y0, size_t ys, const int16_t* dc_y,
                  const int16_t* dc_cb, const int16_t* dc_cr) {
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
    const int16_t* dc = c == 0 ? dc_y : c == 1 ? dc_cb : dc_cr;
    size_t cw = c == 0 ? w : w / 2;
    size_t cx0 = c == 0 ? x0 : x0 / 2;
    size_t cy0 = c == 0 ? y0 : y0 / 2;
    size_t cxs = c == 0 ? xs : xs / 2;
    size_t cys = c == 0 ? ys : ys / 2;

    dc = dc + cy0 * cw + cx0;

    for (size_t y = 0; y < cys; y++) {
      for (size_t x = 0; x < cxs; x++) {
        int16_t px = dc[y * cw + x];
        int16_t left = x ? dc[y * cw + x - 1] : y ? dc[(y - 1) * cw + x] : 0;
        int16_t top = y ? dc[(y - 1) * cw + x] : left;
        int16_t topleft = x && y ? dc[(y - 1) * cw + x - 1] : left;
        int16_t ac = left - topleft;
        int16_t ab = left - top;
        int16_t bc = top - topleft;
        int16_t grad = static_cast<int16_t>(static_cast<uint16_t>(ac) +
                                            static_cast<uint16_t>(top));
        int16_t d = ab ^ bc;
        int16_t clamp = d < 0 ? top : left;
        int16_t s = ac ^ bc;
        int16_t pred = s < 0 ? grad : clamp;

        EncodeU32(PackSigned(px - pred), codes[c], writer, "DC", c, is_delta);
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
  FastMJXLEncoder(size_t w, size_t h,
                  void (*run_on_threads)(void*,
                                         void (*task)(void*, size_t shard),
                                         void*, size_t count),
                  void* thread_runner_data)
      : w(w),
        h(h),
        run_on_threads(run_on_threads),
        thread_runner_data(thread_runner_data) {
    assert(w > 256 || h > 256);
    assert(w % 16 == 0);
    assert(h % 16 == 0);
    prev_ydct.reset(static_cast<int16_t*>(malloc(w * h * sizeof(int16_t))));
    prev_cbdct.reset(
        static_cast<int16_t*>(malloc(w * h * sizeof(int16_t) / 4)));
    prev_crdct.reset(
        static_cast<int16_t*>(malloc(w * h * sizeof(int16_t) / 4)));

    dc_y.reset(static_cast<int16_t*>(malloc(w * h * sizeof(int16_t) / 64)));
    dc_cb.reset(static_cast<int16_t*>(malloc(w * h * sizeof(int16_t) / 256)));
    dc_cr.reset(static_cast<int16_t*>(malloc(w * h * sizeof(int16_t) / 256)));

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
    AllocateEncoded();

    // TODO(veluca): more sensible prefix codes.
    uint64_t nnz_counts[2][3][16] = {
        {{177275, 12727, 5051, 763, 29, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {3739, 6057, 31475, 129745, 291193, 250743, 70415, 0, 0, 0, 0, 0, 0, 0,
          0, 0},
         {137551, 33873, 20637, 3697, 87, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {{674817, 57525, 40719, 9813, 491, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {6477, 1623, 16315, 157505, 840013, 1598263, 513251, 0, 0, 0, 0, 0, 0,
          0, 0, 0},
         {390493, 174537, 150723, 64875, 2737, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
          0}}};
    uint64_t ac_counts[2][3][16] = {
        {{40795, 11277, 15779, 809, 25, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0},
         {18534849, 4243219, 5038379, 1550255, 800155, 382091, 121945, 23077,
          3051, 297, 1, 0, 0, 0, 0, 0},
         {180205, 48335, 49013, 1291, 79, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0}},
        {{402585, 82053, 115183, 4409, 327, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0},
         {76143553, 20669389, 27737567, 9438691, 4293707, 2278787, 894093,
          208333, 29863, 3195, 35, 0, 0, 0, 0, 0},
         {1678387, 358637, 493099, 12599, 555, 11, 1, 1, 1, 1, 1, 0, 0, 0, 0,
          0}}};
    uint64_t dc_counts[2][3][16] = {
        {{109251, 92757, 154085, 155425, 85375, 56825, 59299, 47107, 18847,
          3951, 449, 0, 0, 0, 0, 0},
         {98691, 35591, 42591, 14305, 3961, 627, 79, 3, 1, 1, 1, 0, 0, 0, 0, 0},
         {91527, 41027, 49029, 12497, 1597, 167, 3, 1, 1, 1, 1, 0, 0, 0, 0, 0}},
        {{345091, 308721, 548475, 669509, 437125, 235159, 222595, 210649,
          120651, 30423, 4761, 292, 0, 0, 0, 0},
         {367055, 149617, 182073, 65967, 15929, 2319, 389, 19, 1, 1, 1, 0, 0, 0,
          0, 0},
         {312839, 171951, 216303, 74307, 7381, 549, 37, 1, 1, 1, 1, 0, 0, 0, 0,
          0}}};

    for (size_t j = 0; j < 2; j++) {
      for (size_t i = 0; i < 3; i++) {
        prefix_codes.ac_codes[j][i] = PrefixCode(&ac_counts[j][i][0]);
        prefix_codes.nnz_codes[j][i] = PrefixCode(&nnz_counts[j][i][0]);
        prefix_codes.dc_codes[j][i] = PrefixCode(&dc_counts[j][i][0]);
      }
    }
  }

  void AllocateEncoded() { encoded.Allocate(w * h * 32 + 1024); }

  void AddYCbCrP010Frame(const uint8_t* y_plane, const uint8_t* uv_plane,
                         bool is_last) {
    for (size_t i = 0; i < group_data.size(); i++) {
      group_data[i].Rewind();
    }
    encoded.Rewind();

    if (frame_count == 0) {
      AddImageHeader(&encoded, w, h);
    }

    constexpr size_t kKeyFrameGap = 1;

    bool is_delta = frame_count % kKeyFrameGap != 0;
    bool store_for_next =
        !is_last && (frame_count % kKeyFrameGap != kKeyFrameGap - 1);

    StoreDCGlobal(group_data.data(), is_delta, prefix_codes);
    size_t acg_off = 2 + num_dc_groups_x * num_dc_groups_y;
    StoreACGlobal(group_data.data() + acg_off - 1, num_groups_x * num_groups_y,
                  is_delta, prefix_codes);

    auto process_one_ac = [&](size_t i) {
      size_t ix = i % num_groups_x;
      size_t iy = i / num_groups_x;
      size_t x0 = ix * 256;
      size_t y0 = iy * 256;
      size_t xs = std::min(w, x0 + 256) - x0;
      size_t ys = std::min(h, y0 + 256) - y0;
      if (is_delta) {
        StoreACGroup<true>(store_for_next, group_data.data() + acg_off + i,
                           prefix_codes, w, y_plane, uv_plane, x0, xs, y0, ys,
                           dc_y.get(), prev_ydct.get(), dc_cb.get(),
                           prev_cbdct.get(), dc_cr.get(), prev_crdct.get());
      } else {
        StoreACGroup<false>(store_for_next, group_data.data() + acg_off + i,
                            prefix_codes, w, y_plane, uv_plane, x0, xs, y0, ys,
                            dc_y.get(), prev_ydct.get(), dc_cb.get(),
                            prev_cbdct.get(), dc_cr.get(), prev_crdct.get());
      }
    };

    run_on_threads(
        thread_runner_data,
        +[](void* lambda, size_t i) {
          (*static_cast<decltype(&process_one_ac)>(lambda))(i);
        },
        &process_one_ac, num_groups_x * num_groups_y);

    auto process_one_dc = [&](size_t i) {
      size_t ix = i % num_dc_groups_x;
      size_t iy = i / num_dc_groups_x;
      size_t x0 = ix * 256;
      size_t y0 = iy * 256;
      size_t xs = std::min(w / 8, x0 + 256) - x0;
      size_t ys = std::min(h / 8, y0 + 256) - y0;
      StoreDCGroup(group_data.data() + 1 + i, is_delta, prefix_codes, w, x0, xs,
                   y0, ys, dc_y.get(), dc_cb.get(), dc_cr.get());
    };

    run_on_threads(
        thread_runner_data,
        +[](void* lambda, size_t i) {
          (*static_cast<decltype(&process_one_dc)>(lambda))(i);
        },
        &process_one_dc, num_dc_groups_x * num_dc_groups_y);

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

    std::vector<size_t> group_offsets(group_data.size() + 1);

    for (size_t i = 0; i < group_data.size(); i++) {
      group_data[i].ZeroPadToByte();
      assert(group_data[i].bits_in_buffer == 0);
      size_t sz = group_data[i].bytes_written;
      group_offsets[i + 1] = sz + group_offsets[i];
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

    auto join_groups = [&](size_t i) {
      memcpy(encoded.data.get() + encoded.bytes_written + group_offsets[i],
             group_data[i].data.get(), group_data[i].bytes_written);
    };

    run_on_threads(
        thread_runner_data,
        +[](void* lambda, size_t i) {
          (*static_cast<decltype(&join_groups)>(lambda))(i);
        },
        &join_groups, group_data.size());

    encoded.bytes_written += group_offsets.back();

    frame_count++;
  }

  size_t w;
  size_t h;
  void (*run_on_threads)(void*, void (*task)(void*, size_t shard), void*,
                         size_t count);
  void* thread_runner_data;
  BitWriter encoded;
  size_t encoded_size = 0;
  size_t frame_count = 0;
  DataBuf<int16_t> prev_ydct = {nullptr, free};
  DataBuf<int16_t> prev_cbdct = {nullptr, free};
  DataBuf<int16_t> prev_crdct = {nullptr, free};

  DataBuf<int16_t> dc_y = {nullptr, free};
  DataBuf<int16_t> dc_cb = {nullptr, free};
  DataBuf<int16_t> dc_cr = {nullptr, free};

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
struct FastMJXLEncoder* FastMJXLCreateEncoder(
    size_t width, size_t height,
    void (*run_on_threads)(void*, void (*task)(void*, size_t shard), void*,
                           size_t count),
    void* thread_runner_data) {
  return new FastMJXLEncoder(width, height, run_on_threads, thread_runner_data);
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

uint8_t* FastMJXLReleaseOutputBuffer(struct FastMJXLEncoder* encoder) {
  uint8_t* out = encoder->encoded.data.release();
  encoder->AllocateEncoded();
  assert(encoder->encoded.bits_in_buffer == 0);
  return out;
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
