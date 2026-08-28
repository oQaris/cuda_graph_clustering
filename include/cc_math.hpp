// Objective function of non-strict k-correlation clustering and the algebra of
// a single-vertex move. Everything here compiles for both host and device, so
// the GPU kernels and the CPU reference use literally the same arithmetic.
//
// Notation
//   n        number of vertices
//   k        upper bound on the number of clusters (non-strict: empty ones are
//            allowed, so any labelling with labels in [0, k) is feasible)
//   m        number of edges
//   labels   labels[v] in [0, k)
//   sizes    sizes[c] = number of vertices with label c
//   G        G[v][c] = number of neighbours of v that carry label c   (G = A*Z)
//   W        number of intra-cluster edges = tr(Z^T A Z) / 2
//
// A disagreement is a pair of vertices that is either joined and split apart, or
// unjoined and put together. Counting over pairs i < j:
//
//   f = (same-cluster pairs - intra edges) + (edges - intra edges)
//     = m + sum_c C(sizes[c], 2) - 2 * W
//
// A move of vertex v from cluster a to cluster c changes the two variable terms
// by an amount that only depends on two cluster sizes and two entries of G.
#pragma once

#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#if defined(__CUDACC__)
#define CC_HD __host__ __device__
#else
#define CC_HD
#endif

namespace cc {

// ---------------------------------------------------------------------------
// Host-side bit helpers. The device uses __popc directly; MSVC has neither of
// the GCC/Clang builtins, so both spellings live here rather than at the call
// sites.
// ---------------------------------------------------------------------------

inline int HostPopCount(uint32_t x) {
#if defined(_MSC_VER)
  return static_cast<int>(__popcnt(x));
#else
  return __builtin_popcount(x);
#endif
}

// Index of the lowest set bit. Undefined for x == 0, like the builtins.
inline unsigned HostLowestSetBit(uint32_t x) {
#if defined(_MSC_VER)
  unsigned long index = 0;
  _BitScanForward(&index, x);
  return static_cast<unsigned>(index);
#else
  return static_cast<unsigned>(__builtin_ctz(x));
#endif
}

// Number of unordered pairs inside the clusters: sum_c C(sizes[c], 2).
CC_HD inline long long SamePairs(const int* sizes, int k) {
  long long acc = 0;
  for (int c = 0; c < k; ++c) {
    const long long s = sizes[c];
    acc += s * (s - 1) / 2;
  }
  return acc;
}

// f = m + sum_c C(sizes[c], 2) - 2 * intra_edges
CC_HD inline long long Objective(long long edges, const int* sizes, int k,
                                 long long intra_edges) {
  return edges + SamePairs(sizes, k) - 2 * intra_edges;
}

// Change of f when one vertex leaves cluster `from` (of size size_from,
// contributing g_from neighbours) and joins cluster `to` (size size_to,
// g_to neighbours). Negative means improvement.
//
//   d(same pairs) = size_to - (size_from - 1)
//   d(intra edges) = g_to - g_from
//   d(f) = d(same pairs) - 2 * d(intra edges)
CC_HD inline int MoveDelta(int size_from, int size_to, int g_from, int g_to) {
  return (size_to - size_from + 1) - 2 * (g_to - g_from);
}

// ---------------------------------------------------------------------------
// Deterministic counter-based RNG (splitmix64). Stateless apart from the seed
// the caller carries, which makes it usable inside a kernel without a
// per-thread state array and keeps runs reproducible on both host and device.
// ---------------------------------------------------------------------------

CC_HD inline uint64_t SplitMix64(uint64_t& state) {
  state += 0x9E3779B97F4A7C15ull;
  uint64_t z = state;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

CC_HD inline uint32_t NextU32(uint64_t& state) {
  return static_cast<uint32_t>(SplitMix64(state) >> 32);
}

// Uniform value in [0, bound), unbiased enough for search randomisation.
CC_HD inline unsigned RandBelow(uint64_t& state, unsigned bound) {
  if (bound == 0) return 0;
  return static_cast<unsigned>(
      (static_cast<uint64_t>(NextU32(state)) * bound) >> 32);
}

// Uniform float in [0, 1).
CC_HD inline float RandFloat(uint64_t& state) {
  return static_cast<float>(NextU32(state) >> 8) * (1.0f / 16777216.0f);
}

// Distinct seeds for independent streams (population member, iteration, ...).
CC_HD inline uint64_t StreamSeed(uint64_t seed, uint64_t stream, uint64_t step) {
  uint64_t s = seed ^ (stream * 0xD1B54A32D192ED03ull) ^
               (step * 0xA0761D6478BD642Full);
  SplitMix64(s);
  return s;
}

}  // namespace cc
