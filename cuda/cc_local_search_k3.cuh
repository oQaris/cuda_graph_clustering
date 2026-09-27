// Спуск при k = 3 на широком пути. Отдельное ядро, а не общее: состояние вершины помещается в одно 64-битное слово,
// и за ход ядро читает вдвое меньше памяти, чем общее по G и меткам.
#pragma once

#include "cc_gpu_common.cuh"
#include "cc_gpu_tuning.cuh"
#include "cc_graph.hpp"

namespace cc::gpu {

// Слово вершины: G[0], G[1], G[2] по 20 бит (G <= n - 1 < 2^20) и метка в двух старших битах — сдвиг сразу даёт её
// без маски. Перенос соседа из from в to правит слово одним сложением: G[from] >= 1 (там сам перенесённый), а G[to] + 1
// не больше степени, так что переносов между полями нет.
constexpr int kK3FieldBits = BitWidth(kMaxVerticesWide - 1);
constexpr unsigned long long kK3FieldMask = (1ull << kK3FieldBits) - 1;
constexpr int kK3LabelShift = 62;
constexpr unsigned long long kK3LabelMask = 3ull << kK3LabelShift;
static_assert(3 * kK3FieldBits <= kK3LabelShift, "three G fields and the label must fit one word");

__device__ __forceinline__ int K3Field(unsigned long long word, int c) {
  return (int)((word >> (c * kK3FieldBits)) & kK3FieldMask);
}

// Слова лежат в отдельном буфере work (8n байт на особь), G и метки восстанавливаются в конце. Лучшая цель вершины —
// меньший h из двух чужих кластеров, при равенстве меньший номер:
//   Δ(a -> c) = h_c - h_a + 1,  где h_c = s_c - 2 G[c].
// Размеры кластеров каждый поток ведёт в регистрах: выбранный ход видят все. Сетка постоянная, как у ядра k = 2.
template <int kThreads>
__global__ void __launch_bounds__(kThreads)
    KernelLocalSearch3Wide(const uint32_t* __restrict__ bits, int words, int n, int population, int* all_labels,
                           WideGain* all_g, int* all_sizes, long long* f, unsigned long long* work,
                           unsigned long long* moves, int* next_individual) {
  using Key = MoveKey<unsigned long long>;
  using Move = PackedMove<BitWidth(2)>;
  constexpr int kWarps = kThreads / kWarpSize;

  const int tid = threadIdx.x;
  const uint32_t lane_bit = 1u << (threadIdx.x % kWordBits);
  __shared__ unsigned long long warp_best[2][kWarps];

  for (int sol = ClaimIndividual(next_individual); sol < population; sol = ClaimIndividual(next_individual)) {
    int* labels = all_labels + (size_t)sol * n;
    WideGain* g = all_g + (size_t)sol * 3 * n;
    int* sizes = all_sizes + (size_t)sol * 3;
    unsigned long long* state = work + (size_t)sol * n;
    int s0 = sizes[0];
    int s1 = sizes[1];
    int s2 = sizes[2];
    long long objective = f[sol];

    int best_delta = 0;
    int best_move = -1;
    auto consider = [&](unsigned long long word, int u) {
      const int a = (int)(word >> kK3LabelShift);
      const int h0 = s0 - 2 * K3Field(word, 0);
      const int h1 = s1 - 2 * K3Field(word, 1);
      const int h2 = s2 - 2 * K3Field(word, 2);
      const int ha = a == 0 ? h0 : (a == 1 ? h1 : h2);
      const int low = a == 0 ? h1 : h0;  // чужой кластер с меньшим номером
      const int high = a == 2 ? h1 : h2;
      const bool take_high = high < low;
      const int delta = (take_high ? high : low) - ha + 1;
      if (delta < best_delta) {
        best_delta = delta;
        best_move = (int)Move::Pack(u, a, take_high ? (a == 2 ? 1 : 2) : (a == 0 ? 1 : 0));
      }
    };
    auto key_of = [&]() { return best_move < 0 ? Key::kNone : Key::Pack(best_delta, (unsigned)best_move); };

    for (int u = tid; u < n; u += kThreads) {
      const unsigned long long word = (unsigned long long)g[u] | (unsigned long long)g[n + u] << kK3FieldBits |
                                      (unsigned long long)g[2 * n + u] << (2 * kK3FieldBits) |
                                      (unsigned long long)labels[u] << kK3LabelShift;
      state[u] = word;
      consider(word, u);
    }
    unsigned long long key = key_of();

    unsigned long long applied = 0;
    int parity = 0;
    while (true) {
      const unsigned long long best = BlockMin<kWarps>(key, warp_best[parity]);
      if (best == Key::kNone) break;
      parity ^= 1;

      const unsigned move = Key::Move(best);
      const int v = Move::Vertex(move);
      const int from = Move::From(move);
      const int to = Move::To(move);
      s0 += (to == 0) - (from == 0);
      s1 += (to == 1) - (from == 1);
      s2 += (to == 2) - (from == 2);
      const unsigned long long inc = (1ull << (kK3FieldBits * to)) - (1ull << (kK3FieldBits * from));
      if (v % kThreads == tid) state[v] = (state[v] & ~kK3LabelMask) | (unsigned long long)to << kK3LabelShift;

      const uint32_t* row = bits + (size_t)v * words;
      best_delta = 0;
      best_move = -1;
#pragma unroll(tuning::kWideUnroll)
      for (int u = tid; u < n; u += kThreads) {
        unsigned long long word = state[u];
        if (__ldg(row + u / kWordBits) & lane_bit) {
          word += inc;
          state[u] = word;
        }
        consider(word, u);
      }
      key = key_of();
      objective += Key::Delta(best);
      ++applied;
    }

    for (int u = tid; u < n; u += kThreads) {
      const unsigned long long word = state[u];
      g[u] = K3Field(word, 0);
      g[n + u] = K3Field(word, 1);
      g[2 * n + u] = K3Field(word, 2);
      labels[u] = (int)(word >> kK3LabelShift);
    }
    if (tid == 0) {
      sizes[0] = s0;
      sizes[1] = s1;
      sizes[2] = s2;
    }
    StoreDescent(&f[sol], objective, moves, applied);
  }
}

}  // namespace cc::gpu
