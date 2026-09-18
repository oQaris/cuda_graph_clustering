// Целевая функция нестрогой k-корреляционной кластеризации и алгебра одиночного хода. Файл
// компилируется и для хоста, и для устройства, поэтому CPU и GPU используют буквально одну и ту
// же арифметику.
//
// Обозначения:
//   n      число вершин
//   k      верхняя граница числа кластеров (нестрого: пустые кластеры допустимы,
//          годится любая расстановка меток из [0, k))
//   m      число рёбер
//   labels labels[v] из [0, k)
//   sizes  sizes[c] = число вершин с меткой c
//   G      G[v][c] = число соседей v с меткой c   (G = A*Z)
//   W      число внутрикластерных рёбер = tr(Z^T A Z) / 2
//
// Разногласие — пара вершин, которая либо соединена и разнесена по кластерам, либо не соединена и
// оказалась в одном. Считая по парам i < j:
//
//   f = (пар в одном кластере − внутренних рёбер) + (рёбер − внутренних рёбер)
//     = m + sum_c C(sizes[c], 2) − 2 * W
//
// Перенос вершины v из кластера a в кластер c меняет оба переменных слагаемых на величину,
// зависящую только от двух размеров кластеров и двух элементов G.
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

// Битовые примитивы для хоста: на устройстве используется __popc напрямую, а у MSVC нет билтинов
// GCC/Clang, поэтому обе версии живут здесь, а не в местах вызова.

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

// Изменение f при переносе вершины из кластера from (размер size_from, g_from соседей в нём) в
// кластер to (size_to, g_to соседей). Отрицательное значение — улучшение.
//
//   d(пар в одном кластере) = size_to - (size_from - 1)
//   d(внутренних рёбер)     = g_to - g_from
//   d(f) = d(пар) - 2 * d(рёбер)
CC_HD inline int MoveDelta(int size_from, int size_to, int g_from, int g_to) {
  return (size_to - size_from + 1) - 2 * (g_to - g_from);
}

// ---------------------------------------------------------------------------
// Детерминированный счётчиковый ГПСЧ (splitmix64). Без внутреннего состояния, кроме сида, который
// несёт с собой вызывающий код — это позволяет использовать генератор внутри ядра без массива
// состояний на поток и делает прогоны воспроизводимыми и на хосте, и на устройстве.
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

// Равномерное значение в [0, bound).
CC_HD inline unsigned RandBelow(uint64_t& state, unsigned bound) {
  if (bound == 0) return 0;
  return static_cast<unsigned>((static_cast<uint64_t>(NextU32(state)) * bound) >> 32);
}

// Равномерное вещественное в [0, 1).
CC_HD inline float RandFloat(uint64_t& state) {
  return static_cast<float>(NextU32(state) >> 8) * (1.0f / 16777216.0f);
}

// Независимые потоки ГПСЧ (особь популяции, итерация, ...) из одного сида.
CC_HD inline uint64_t StreamSeed(uint64_t seed, uint64_t stream, uint64_t step) {
  uint64_t s = seed ^ (stream * 0xD1B54A32D192ED03ull) ^ (step * 0xA0761D6478BD642Full);
  SplitMix64(s);
  return s;
}

}  // namespace cc
