// Целевая функция нестрогой k-корреляционной кластеризации, алгебра одиночного хода и ГПСЧ. Файл компилируется и для
// хоста, и для устройства, так что CPU и GPU используют буквально одну арифметику.
//
// Обозначения:
//   n      число вершин
//   k      верхняя граница числа кластеров (нестрого: пустые кластеры допустимы, годится любая метка из [0, k))
//   m      число рёбер
//   sizes  sizes[c] = число вершин с меткой c
//   G      G[v][c] = число соседей v с меткой c   (G = A*Z)
//   W      число внутрикластерных рёбер = tr(Z^T A Z) / 2
//
// Разногласие — пара вершин, которая соединена и разнесена по кластерам или не соединена и оказалась в одном. По парам
// i < j:
//
//   f = (пар в одном кластере − внутренних рёбер) + (рёбер − внутренних рёбер) = m + sum_c C(sizes[c], 2) − 2 * W
//
// Перенос вершины из кластера a в c меняет оба переменных слагаемых на величину, зависящую только от двух размеров
// кластеров и двух элементов G.
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

// Битовые примитивы хоста: на устройстве есть __popc, а у MSVC нет билтинов GCC/Clang.
inline int HostPopCount(uint32_t x) {
#if defined(_MSC_VER)
  return static_cast<int>(__popcnt(x));
#else
  return __builtin_popcount(x);
#endif
}

// Номер младшего установленного бита. Не определено при x == 0, как и у билтинов.
inline unsigned HostLowestSetBit(uint32_t x) {
#if defined(_MSC_VER)
  unsigned long index = 0;
  _BitScanForward(&index, x);
  return static_cast<unsigned>(index);
#else
  return static_cast<unsigned>(__builtin_ctz(x));
#endif
}

// Число пар внутри кластеров: sum_c C(sizes[c], 2).
CC_HD inline long long SamePairs(const int* sizes, int k) {
  long long acc = 0;
  for (int c = 0; c < k; ++c) {
    const long long s = sizes[c];
    acc += s * (s - 1) / 2;
  }
  return acc;
}

// f = m + sum_c C(sizes[c], 2) - 2 * intra_edges
CC_HD inline long long Objective(long long edges, const int* sizes, int k, long long intra_edges) {
  return edges + SamePairs(sizes, k) - 2 * intra_edges;
}

// Изменение f при переносе вершины из кластера from (размер size_from, в нём g_from её соседей) в кластер to (size_to,
// g_to соседей). Отрицательное — улучшение.
//   d(пар в одном кластере) = size_to - (size_from - 1)
//   d(внутренних рёбер)     = g_to - g_from
//   d(f) = d(пар) - 2 * d(рёбер)
CC_HD inline int MoveDelta(int size_from, int size_to, int g_from, int g_to) {
  return (size_to - size_from + 1) - 2 * (g_to - g_from);
}

// ---------------------------------------------------------------------------------------------------------------------
// Детерминированный счётчиковый ГПСЧ (splitmix64). Всё его состояние — сид, который несёт вызывающий код: генератор
// работает в ядре без массива состояний на поток, а прогоны воспроизводимы и на хосте, и на устройстве.
// ---------------------------------------------------------------------------------------------------------------------

CC_HD inline uint64_t SplitMix64(uint64_t& state) {
  state += 0x9E3779B97F4A7C15ull;
  uint64_t z = state;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

CC_HD inline uint32_t NextU32(uint64_t& state) { return static_cast<uint32_t>(SplitMix64(state) >> 32); }

// Равномерное значение в [0, bound).
CC_HD inline unsigned RandBelow(uint64_t& state, unsigned bound) {
  if (bound == 0) return 0;
  return static_cast<unsigned>((static_cast<uint64_t>(NextU32(state)) * bound) >> 32);
}

// Равномерное вещественное в [0, 1): старшие 24 бита — ровно мантисса float.
CC_HD inline float RandFloat(uint64_t& state) {
  constexpr int kMantissaBits = 24;
  return static_cast<float>(NextU32(state) >> (32 - kMantissaBits)) * (1.0f / (1 << kMantissaBits));
}

// Независимые потоки ГПСЧ (особь, итерация, ...) из одного сида.
CC_HD inline uint64_t StreamSeed(uint64_t seed, uint64_t stream, uint64_t step) {
  uint64_t s = seed ^ (stream * 0xD1B54A32D192ED03ull) ^ (step * 0xA0761D6478BD642Full);
  SplitMix64(s);
  return s;
}

// ---------------------------------------------------------------------------------------------------------------------
// Случайные решения PBILS. Оба бэкенда выводят сиды этими функциями, поэтому при одном --seed идут одной траекторией.
// base — StreamSeed(seed, k*Stream, 0), index — номер метки в популяции: особь * n + вершина.
// ---------------------------------------------------------------------------------------------------------------------

constexpr uint64_t kInitStream = 0xA11CE;
constexpr uint64_t kSelectStream = 0x5E1EC7;
constexpr uint64_t kPerturbStream = 0xBEEF;
constexpr uint64_t kPerturbFirstStep = 0x5000000;  // смена любой из констант меняет все траектории

CC_HD inline int InitialLabel(uint64_t base, uint64_t index, int k) {
  uint64_t rng = StreamSeed(base, index, 0);
  return static_cast<int>(RandBelow(rng, k));
}

// Победитель турнира за слот следующей популяции: лучшая по f из tournament случайных особей.
CC_HD inline int Tournament(uint64_t base, int slot, int iteration, const long long* f, int population,
                            int tournament) {
  uint64_t rng = StreamSeed(base, slot, iteration + 1);
  int chosen = static_cast<int>(RandBelow(rng, population));
  for (int t = 1; t < tournament; ++t) {
    const int candidate = static_cast<int>(RandBelow(rng, population));
    if (f[candidate] < f[chosen]) chosen = candidate;
  }
  return chosen;
}

// С вероятностью probability — метка равновероятного другого кластера, иначе прежняя.
CC_HD inline int PerturbedLabel(uint64_t base, uint64_t index, int iteration, int label, int k, float probability) {
  uint64_t rng = StreamSeed(base, index, iteration + kPerturbFirstStep);
  if (RandFloat(rng) >= probability) return label;
  return (label + 1 + static_cast<int>(RandBelow(rng, k - 1))) % k;
}

}  // namespace cc
