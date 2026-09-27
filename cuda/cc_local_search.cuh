// Общее ядро спуска: годится для любого k и обоих путей.
#pragma once

#include <type_traits>

#include "cc_gpu_common.cuh"
#include "cc_graph.hpp"
#include "cc_math.hpp"

namespace cc::gpu {

// Наискорейший спуск по одиночным переносам вершин до локального оптимума, блок на особь, один барьер на ход.
// kShared: G и метки на время спуска копируются в разделяемую память (2kn + n байт, хост проверяет заранее), иначе
// читаются прямо из глобальной. Оба варианта проходят одну траекторию.
template <bool kShared, int kThreads, typename GainT>
__global__ void __launch_bounds__(kThreads)
    KernelLocalSearch(const uint32_t* __restrict__ bits, int words, int n, int k, int* all_labels, GainT* all_g,
                      int* all_sizes, long long* f, unsigned long long* moves) {
  using Label = std::conditional_t<kShared, signed char, int>;  // байт на метку: в разделяемую влезает больший граф
  using Key = MoveKey<unsigned long long>;
  using Move = PackedMove<BitWidth(kMaxClusters - 1)>;
  constexpr int kWarps = kThreads / kWarpSize;

  const int sol = blockIdx.x;
  const int tid = threadIdx.x;
  int* labels = all_labels + (size_t)sol * n;
  GainT* g = all_g + (size_t)sol * k * n;
  int* sizes = all_sizes + (size_t)sol * k;

  // Поток владеет вершинами v = tid (mod kThreads) и в копировании, и в скане, и в правке G: читает и пишет только
  // свои элементы, поэтому барьеры после копирования и после хода не нужны.
  GainT* wg;
  Label* wl;
  if constexpr (kShared) {
    wg = DynamicShared<GainT>();
    wl = reinterpret_cast<Label*>(wg + (size_t)k * n);
    for (int c = 0; c < k; ++c) {
      for (int v = tid; v < n; v += kThreads) wg[(size_t)c * n + v] = g[(size_t)c * n + v];
    }
    for (int v = tid; v < n; v += kThreads) wl[v] = (Label)labels[v];
  } else {
    wg = g;
    wl = labels;
  }

  // Размеры кластеров — двойной буфер: в скане хода i буфер i & 1 хранит размеры до хода i - 1, а поправку за сам
  // ход i - 1 поток берёт из регистров. Массив на поток ушёл бы в стек: при k >= 16 это до 15 % времени (замер).
  __shared__ int cluster_size[2][kMaxClusters];
  __shared__ unsigned long long warp_best[2][kWarps];
  for (int c = tid; c < k; c += kThreads) cluster_size[0][c] = cluster_size[1][c] = sizes[c];
  __syncthreads();

  long long objective = f[sol];
  unsigned long long applied = 0;
  int parity = 0;
  int prev_from = -1;  // последний ход, ещё не учтённый в буфере parity
  int prev_to = -1;
  while (true) {
    const int* sizes_now = cluster_size[parity];
    auto size_of = [&](int c) { return sizes_now[c] + (c == prev_to) - (c == prev_from); };

    int best_delta = 0;
    int best_v = -1;
    int best_from = 0;
    int best_to = 0;
    for (int v = tid; v < n; v += kThreads) {
      const int from = wl[v];
      const int size_from = size_of(from);
      const int g_from = wg[from * n + v];
      for (int c = 0; c < k; ++c) {
        if (c == from) continue;
        const int delta = MoveDelta(size_from, size_of(c), g_from, wg[c * n + v]);
        if (delta < best_delta) {
          best_delta = delta;
          best_v = v;
          best_from = from;
          best_to = c;
        }
      }
    }
    const unsigned long long key =
        best_v < 0 ? Key::kNone : Key::Pack(best_delta, Move::Pack(best_v, best_from, best_to));
    const unsigned long long best = BlockMin<kWarps>(key, warp_best[parity]);
    if (best == Key::kNone) break;

    const unsigned move = Key::Move(best);
    const int v = Move::Vertex(move);
    const int from = Move::From(move);
    const int to = Move::To(move);
    // Буфер parity все уже дочитали (прошли барьер): доводим его до размеров после этого хода.
    for (int c = tid; c < k; c += kThreads) {
      cluster_size[parity][c] += (c == prev_to) - (c == prev_from) + (c == to) - (c == from);
    }
    prev_from = from;
    prev_to = to;
    parity ^= 1;

    const uint32_t* row = bits + (size_t)v * words;
    GainT* g_from = wg + from * n;
    GainT* g_to = wg + to * n;
    for (int u = tid; u < n; u += kThreads) {
      if ((row[u / kWordBits] >> (u % kWordBits)) & 1u) {
        g_from[u] -= 1;
        g_to[u] += 1;
      }
    }
    if (v % kThreads == tid) wl[v] = (Label)to;
    objective += Key::Delta(best);
    ++applied;
  }

  if constexpr (kShared) {
    for (int c = 0; c < k; ++c) {
      for (int v = tid; v < n; v += kThreads) g[(size_t)c * n + v] = wg[(size_t)c * n + v];
    }
    for (int v = tid; v < n; v += kThreads) labels[v] = wl[v];
  }
  // Буфер parity после выхода никто не пишет: остаётся прибавить к нему последний ход.
  for (int c = tid; c < k; c += kThreads) sizes[c] = cluster_size[parity][c] + (c == prev_to) - (c == prev_from);
  StoreDescent(&f[sol], objective, moves, applied);
}

}  // namespace cc::gpu
