// Параметры запуска ядер. Сняты замером на RTX 4070 Ti SUPER (66 SM, 48 МБ L2), данные — в docs/benchmarks.md.
// Шаблонные параметры ядер должны быть известны при сборке, поэтому это заголовок, а не файл настроек.
#pragma once

#include <type_traits>
#include <utility>

#include "cc_gpu_common.cuh"

namespace cc::gpu::tuning {

// Вспомогательные ядра: начальные метки, пересчёт G, селекция, возмущение.
constexpr int kBlockSize = 256;
// Поэлементные ядра идут сеткой с шагом, и ГПСЧ метки от сетки не зависит, так что её размер можно ограничить.
constexpr long long kMaxGridBlocks = 1 << 16;

// Широкий путь: на поток приходятся сотни вершин, и 1024 потока быстрее 512 в 1,1–1,3 раза.
constexpr int kWideThreads = 1024;
constexpr int kWideUnroll = 4;  // развёртка цикла по вершинам в ядрах k = 2 и k = 3

// Ядра k = 2 и k = 3 широкого пути идут постоянной сеткой: блоков столько, чтобы состояние их особей занимало эту долю
// L2. Меньше блоков — простаивают SM, больше — ход идёт из видеопамяти, итерация дороже на 10–25 %. Лучшая доля:
// k = 2 — 0,52–0,74, k = 3 — 0,72–1,1 (разница в пределах 3 %).
constexpr double kL2ShareK2 = 0.6;
constexpr double kL2ShareK3 = 0.8;

// Общее ядро спуска. Пока блоков мало, больший блок быстрее (до 1,95x при популяции 64). Когда популяция заполняет
// SM, а работы на особь мало, быстрее малый (до 1,6x): у большого по несколько вершин на поток, и ход съедает барьер.
constexpr int kGenericSmallBlock = 256;
constexpr int kGenericLargeBlock = 512;
constexpr int kFullPopulation = 256;
constexpr long long kLightWork = 10000;  // k * n

template <typename GainT>
using GenericThreads = std::conditional_t<std::is_same_v<GainT, WideGain>, std::integer_sequence<int, kWideThreads>,
                                          std::integer_sequence<int, kGenericSmallBlock, kGenericLargeBlock>>;

template <typename GainT>
int GenericThreadsFor(int n, int k, int population) {
  if constexpr (std::is_same_v<GainT, WideGain>) {
    return kWideThreads;
  } else {
    return population >= kFullPopulation && (long long)k * n < kLightWork ? kGenericSmallBlock : kGenericLargeBlock;
  }
}

// Ядро k = 2 в регистрах: наименьший блок, в котором на поток не больше kK2MaxVerticesPerThread вершин. Узкое место —
// инструкции на вершину, лишние потоки только удлиняют свёртку, а 16 вершин на поток упираются в регистры (в 2–4
// раза медленнее 8). Выше 12 288 вершин блок в 1024 потока берёт больше: небольшой вылет в стек, но всё равно быстрее.
using K2Threads = std::integer_sequence<int, 128, 256, 512, 1024>;
// Каждое число вершин на поток — свой экземпляр ядра, поэтому здесь не все числа подряд.
using K2VerticesPerThread = std::integer_sequence<int, 1, 2, 4, 8, 12, 16, 24, 32>;
constexpr int kK2MaxVerticesPerThread = 12;
// Исключение: при малой популяции на графе 513–1024 вершин карта недогружена, и 256 потоков быстрее 128 до 1,7x.
constexpr int kK2FewBlocks = 256;
constexpr int kK2FewBlocksMinN = 513;
constexpr int kK2FewBlocksMaxN = 1024;
constexpr int kK2FewBlocksThreads = 256;

static_assert(ValuesOf(K2Threads{}).back() * ValuesOf(K2VerticesPerThread{}).back() >= kMaxVerticesNarrow,
              "k = 2 register kernel must cover the whole narrow path");

inline int K2ThreadsFor(int n, int population) {
  if (population < kK2FewBlocks && n >= kK2FewBlocksMinN && n <= kK2FewBlocksMaxN) return kK2FewBlocksThreads;
  for (const int threads : ValuesOf(K2Threads{})) {
    if (n <= kK2MaxVerticesPerThread * threads) return threads;
  }
  return ValuesOf(K2Threads{}).back();
}

// Наименьшее собранное число вершин на поток, при котором блок покрывает граф.
inline int K2VerticesPerThreadFor(int n, int threads) {
  for (const int vertices : ValuesOf(K2VerticesPerThread{})) {
    if (vertices * threads >= n) return vertices;
  }
  throw std::logic_error("k = 2 register kernel does not cover " + std::to_string(n) + " vertices");
}

}  // namespace cc::gpu::tuning
