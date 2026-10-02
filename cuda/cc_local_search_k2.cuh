// Спуск при k = 2. Отдельные ядра, а не общее: у вершины единственный ход — в другой кластер, и её состояние
// сводится к одному числу e = G[1][v] - G[0][v], которое не зависит от метки:
//   Δf(0 -> 1) = s - 2e,  Δf(1 -> 0) = 2 - s + 2e,  где s = n1 - n0 + 1;
//   перенос любой вершины в кластер 1 прибавляет к e каждого её соседа 2, в кластер 0 — вычитает, а своё e не меняет.
// Так состояние особи помещается в регистры (узкий путь) или в одно слово на вершину (широкий), а ход не читает G.
// В ключе хода вместо пары кластеров — метка вершины: цель однозначна.
#pragma once

#include "cc_gpu_common.cuh"
#include "cc_gpu_tuning.cuh"
#include "cc_graph.hpp"

namespace cc::gpu {

constexpr int kEStep = 2;  // на сколько перенос вершины меняет e её соседа
// Вершины варпа лежат в одном слове строки смежности: загрузка одна на варп, а бит потока в слове — его lane.
static_assert(kWarpSize == (int)kWordBits, "one adjacency word per warp");

// Узкий путь, всё состояние в регистрах. Поток владеет вершинами u = tid + j * kThreads, j < kVpt: e и бит метки
// каждой, а n0 ведёт сам (выбранный ход видят все). За ход память читает только строку смежности v, правка e слита со
// сканом следующего хода.
template <int kThreads, int kVpt>
__global__ void __launch_bounds__(kThreads)
    KernelLocalSearch2(const uint32_t* __restrict__ bits, int words, int n, int* all_labels, Gain* all_g, int* sizes,
                       long long* f, unsigned long long* moves) {
  using Key = MoveKey<unsigned>;
  constexpr int kWarps = kThreads / kWarpSize;
  static_assert(kVpt <= (int)(8 * sizeof(unsigned)), "labels of a thread are packed into one unsigned");
  static_assert(kMaxVerticesNarrow < Key::kOffset && 2 * kMaxVerticesNarrow < (1 << Key::kHalfBits),
                "n and vertex << 1 | label must fit the halves of a 32-bit key");

  const int sol = blockIdx.x;
  const int tid = threadIdx.x;
  const uint32_t lane_bit = 1u << (threadIdx.x % kWordBits);
  int* labels = all_labels + (size_t)sol * n;
  Gain* g = all_g + (size_t)sol * 2 * n;
  __shared__ unsigned warp_best[2][kWarps];

  constexpr int kIdleE = -(1 << 29);  // e вершины за пределами n: её перенос не улучшает f, а правку отсекает active
  int e[kVpt];
  unsigned label = 0;
  unsigned active = 0;
#pragma unroll
  for (int j = 0; j < kVpt; ++j) {
    const int v = tid + j * kThreads;
    e[j] = kIdleE;
    if (v < n) {
      e[j] = (int)g[n + v] - (int)g[v];
      label |= (unsigned)labels[v] << j;
      active |= 1u << j;
    }
  }
  int n0 = sizes[(size_t)sol * 2];
  long long objective = f[sol];

  // Лучший ход потока ищется строгим сравнением по возрастанию j — при равной Δ это меньшая вершина.
  auto key_of = [&](int delta, int j) {
    if (delta >= 0) return Key::kNone;
    return Key::Pack(delta, (unsigned)(tid + j * kThreads) << 1 | ((label >> j) & 1u));
  };
  int best_delta = 0;
  int best_j = 0;
  {
    const int s = n - 2 * n0 + 1;
#pragma unroll
    for (int j = 0; j < kVpt; ++j) {
      const int delta = ((label >> j) & 1u) ? 2 - s + 2 * e[j] : s - 2 * e[j];
      if (delta < best_delta) {
        best_delta = delta;
        best_j = j;
      }
    }
  }
  unsigned key = key_of(best_delta, best_j);

  unsigned long long applied = 0;
  int parity = 0;
  while (true) {
    const unsigned best = BlockMin<kWarps>(key, warp_best[parity]);
    if (best == Key::kNone) break;
    parity ^= 1;

    const unsigned move = Key::Move(best);
    const int v = (int)(move >> 1);
    const unsigned from = move & 1u;
    n0 += from ? 1 : -1;
    const int s = n - 2 * n0 + 1;
    const int step = from ? -kEStep : kEStep;
    if (v % kThreads == tid) label ^= 1u << (v / kThreads);

    const uint32_t* row = bits + (size_t)v * words;
    best_delta = 0;
    best_j = 0;
#pragma unroll
    for (int j = 0; j < kVpt; ++j) {
      // За конец строки выходят только вершины за пределами n: им хватит последнего слова.
      const int word = min((int)((tid + j * kThreads) / kWordBits), words - 1);
      if ((__ldg(row + word) & lane_bit) && ((active >> j) & 1u)) e[j] += step;
      const int delta = ((label >> j) & 1u) ? 2 - s + 2 * e[j] : s - 2 * e[j];
      if (delta < best_delta) {
        best_delta = delta;
        best_j = j;
      }
    }
    key = key_of(best_delta, best_j);
    objective += Key::Delta(best);
    ++applied;
  }

  // G восстанавливается из e и степени: глобальная G за спуск не менялась, а сумма её строк и есть степень.
#pragma unroll
  for (int j = 0; j < kVpt; ++j) {
    const int v = tid + j * kThreads;
    if (v < n) {
      const int degree = (int)g[v] + (int)g[n + v];
      labels[v] = (label >> j) & 1u;
      g[v] = (Gain)((degree - e[j]) / 2);
      g[n + v] = (Gain)((degree + e[j]) / 2);
    }
  }
  if (tid == 0) {
    sizes[(size_t)sol * 2] = n0;
    sizes[(size_t)sol * 2 + 1] = n - n0;
  }
  StoreDescent(&f[sol], objective, moves, applied);
}

// Широкий путь: e и метки в регистры блока уже не помещаются. Состояние вершины — слово w = 2e + метка в глобальной
// памяти на месте строки G[0]: G на время спуска не нужна и в конце восстанавливается из e и степени. Через w
// Δ = s - w при метке 0 и w + 1 - s при метке 1. Сетка постоянная, блоков столько, чтобы состояние (4n байт на
// особь) помещалось в L2.
template <int kThreads>
__global__ void __launch_bounds__(kThreads)
    KernelLocalSearch2Wide(const uint32_t* __restrict__ bits, int words, int n, int population,
                           const int* __restrict__ degrees, int* all_labels, WideGain* all_g, int* sizes, long long* f,
                           unsigned long long* moves, int* next_individual) {
  using Key = MoveKey<unsigned long long>;
  constexpr int kWarps = kThreads / kWarpSize;

  const int tid = threadIdx.x;
  const uint32_t lane_bit = 1u << (threadIdx.x % kWordBits);
  __shared__ unsigned long long warp_best[2][kWarps];

  for (int sol = ClaimIndividual(next_individual); sol < population; sol = ClaimIndividual(next_individual)) {
    int* labels = all_labels + (size_t)sol * n;
    WideGain* g = all_g + (size_t)sol * 2 * n;
    int* state = g;  // строка G[0]

    auto key_of = [&](int delta, int u, int word) {
      return u < 0 ? Key::kNone : Key::Pack(delta, (unsigned)u << 1 | (unsigned)(word & 1));
    };
    int n0 = sizes[(size_t)sol * 2];
    long long objective = f[sol];
    int s = n - 2 * n0 + 1;
    int best_delta = 0;
    int best_u = -1;
    int best_word = 0;
    for (int u = tid; u < n; u += kThreads) {
      const int word = 2 * (g[n + u] - g[u]) + labels[u];
      state[u] = word;
      const int delta = (word & 1) ? word + 1 - s : s - word;
      if (delta < best_delta) {
        best_delta = delta;
        best_u = u;
        best_word = word;
      }
    }
    unsigned long long key = key_of(best_delta, best_u, best_word);

    unsigned long long applied = 0;
    int parity = 0;
    while (true) {
      const unsigned long long best = BlockMin<kWarps>(key, warp_best[parity]);
      if (best == Key::kNone) break;
      parity ^= 1;

      const unsigned move = Key::Move(best);
      const int v = (int)(move >> 1);
      const int from = (int)(move & 1u);
      n0 += from ? 1 : -1;
      s = n - 2 * n0 + 1;
      const int shift = from ? -2 * kEStep : 2 * kEStep;  // w = 2e + метка меняется вдвое сильнее e
      if (v % kThreads == tid) state[v] ^= 1;

      const uint32_t* row = bits + (size_t)v * words;
      best_delta = 0;
      best_u = -1;
#pragma unroll(tuning::kWideUnroll)
      for (int u = tid; u < n; u += kThreads) {
        int word = state[u];
        if (__ldg(row + u / kWordBits) & lane_bit) {
          word += shift;
          state[u] = word;
        }
        const int delta = (word & 1) ? word + 1 - s : s - word;
        if (delta < best_delta) {
          best_delta = delta;
          best_u = u;
          best_word = word;
        }
      }
      key = key_of(best_delta, best_u, best_word);
      objective += Key::Delta(best);
      ++applied;
    }

    for (int u = tid; u < n; u += kThreads) {
      const int word = state[u];
      const int e = word >> 1;
      const int degree = degrees[u];
      labels[u] = word & 1;
      g[u] = (degree - e) / 2;
      g[n + u] = (degree + e) / 2;
    }
    if (tid == 0) {
      sizes[(size_t)sol * 2] = n0;
      sizes[(size_t)sol * 2 + 1] = n - n0;
    }
    StoreDescent(&f[sol], objective, moves, applied);
  }
}

}  // namespace cc::gpu
