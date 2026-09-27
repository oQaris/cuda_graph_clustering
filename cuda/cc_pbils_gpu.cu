// PBILS на CUDA. Полное обоснование схемы — в README ("Устройство GPU-версии"):
// один блок = одна особь популяции, поэтому блоки не синхронизируются между
// собой, а параллелизм даёт сама популяция.
//
// Внутри блока потоки делят между собой вершины, оценивают все n*k ходов по
// матрице G = A*Z, редукцией находят наилучший и правят G за O(n). При n ~ 6000
// это упирается в пропускную способность глобальной памяти (~120 КБ трафика на
// ход), поэтому состояние решения при возможности переносится в разделяемую
// память; G хранится 16-битными счётчиками, что вдвое сокращает трафик и вдвое
// увеличивает инстанс, помещающийся в неё целиком.
//
// На графах больше 32 767 вершин 16 бит не хватает, и решатель идёт широким путём: те же ядра с
// 32-битными G и отдельные ядра спуска для k = 2 и k = 3, у которых состояние вершины упаковано в одно
// слово. Узкий путь от этого не меняется.
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "cc_gpu.hpp"
#include "cc_math.hpp"

namespace cc {
namespace {

// Размер блока: число потоков в одном блоке (blockDim.x), степень двойки — на этом держится
// редукция наилучшего хода и суммы f. Число блоков (gridDim.x) — величина другого рода, она
// зависит от параметров запуска и константой не является: population для ядер "один блок = одна
// особь" (Rebuild, LocalSearch, Select), init_blocks для поэлементных ядер (InitLabels, Perturb).
constexpr int kBlockSize = 256;

// Элементы G — счётчики соседей, не больше n. Пока n не больше kMaxVerticesForGain, хватает 16 бит
// (узкий путь); на большем графе G 32-битная (широкий путь), и предел задаёт упаковка хода.
using Gain = short;
using WideGain = int;
constexpr int kMaxVerticesForGain = 32767;

// Ход в KernelLocalSearch упакован в 32 бита без знака: вершина << 12 | откуда << 6 | куда.
constexpr int kMaxVerticesWide = 1 << 20;
static_assert(kMaxClusters <= 64, "cluster index must fit in 6 bits of the packed move");

#define CC_CUDA_CHECK(call)                                                                    \
  do {                                                                                          \
    const cudaError_t status = (call);                                                          \
    if (status != cudaSuccess) {                                                                \
      throw std::runtime_error(std::string("cuda error at ") + __FILE__ + ":" +                 \
                               std::to_string(__LINE__) + ": " + cudaGetErrorString(status));    \
    }                                                                                            \
  } while (0)

double SecondsSince(const std::chrono::steady_clock::time_point& start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// --------------------------------------------------------------------------
// Ядра
// --------------------------------------------------------------------------

// Поэлементные ядра (InitLabels, Perturb) идут по всем population*n меткам. На узком пути их число
// укладывается в int, на широком может и не уложиться, поэтому тип индекса — параметр шаблона.
template <typename Index>
__global__ void KernelInitLabels(Index total, int k, int* labels, uint64_t seed) {
  const Index stride = blockDim.x * gridDim.x;
  for (Index idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
    uint64_t rng = StreamSeed(seed, idx, 0);
    labels[idx] = RandBelow(rng, k);
  }
}

// Пересчитывает G, размеры кластеров и f по labels. G = A*Z считается битовым параллелизмом:
// строка смежности пересекается с маской каждого кластера и суммируется через __popc — это и
// есть побитовая форма матричного произведения.
//
// В горячем цикле по (v, c) индексы держим в size_t, хотя они и укладываются в int: с
// int-арифметикой ptxas выделяет здесь 40 регистров вместо 48, и ядро на карте становится
// медленнее. Замерено; не убирайте size_t, не повторив замер.
template <typename GainT>
__global__ void KernelRebuild(const uint32_t* __restrict__ bits, int words, int n, int k,
                              const int* __restrict__ labels, GainT* g, int* sizes, long long* f,
                              uint32_t* masks, long long edges) {
  const int sol = blockIdx.x;
  const int tid = threadIdx.x;

  const int* my_labels = labels + (size_t)sol * n;
  GainT* my_g = g + (size_t)sol * k * n;
  int* my_sizes = sizes + (size_t)sol * k;
  uint32_t* my_masks = masks + (size_t)sol * k * words;

  for (int i = tid; i < k * words; i += kBlockSize) my_masks[i] = 0u;
  for (int c = tid; c < k; c += kBlockSize) my_sizes[c] = 0;
  __syncthreads();

  for (int v = tid; v < n; v += kBlockSize) {
    const int c = my_labels[v];
    atomicOr(&my_masks[(size_t)c * words + (v >> 5)], 1u << (v & 31));
    atomicAdd(&my_sizes[c], 1);
  }
  __syncthreads();

  long long intra_twice = 0;
  for (int v = tid; v < n; v += kBlockSize) {
    const uint32_t* row = bits + (size_t)v * words;
    for (int c = 0; c < k; ++c) {
      const uint32_t* mask = my_masks + (size_t)c * words;
      int acc = 0;
      for (int w = 0; w < words; ++w) acc += __popc(row[w] & mask[w]);
      my_g[(size_t)c * n + v] = (GainT)acc;
    }
    intra_twice += my_g[(size_t)my_labels[v] * n + v];
  }

  __shared__ long long reduce[kBlockSize];
  reduce[tid] = intra_twice;
  __syncthreads();
  for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) reduce[tid] += reduce[tid + stride];
    __syncthreads();
  }

  if (tid == 0) f[sol] = Objective(edges, my_sizes, k, reduce[0] / 2);
}

// Степени вершин, один поток на вершину. Нужны KernelRebuild2: при k = 2 G[0][v] = степень - G[1][v].
__global__ void KernelDegrees(const uint32_t* __restrict__ bits, int words, int n, int* degrees) {
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= n) return;
  const uint32_t* row = bits + (size_t)v * words;
  int acc = 0;
  for (int w = 0; w < words; ++w) acc += __popc(row[w]);
  degrees[v] = acc;
}

// KernelRebuild для k = 2. Маска кластера 1 собирается __ballot_sync прямо в разделяемую память
// (варп читает 32 метки подряд — это ровно одно слово маски), и popcount нужен только по ней:
// G[0] достраивается из степени. Выходит вдвое меньше проходов по матрице и ни одного атомика.
// Наибольшее n для ядер k = 2 узкого пути: вершина должна уложиться в 15 бит ключа KernelLocalSearch2.
// На широком пути маска не помещается в статический массив и лежит в динамической разделяемой памяти
// (words * 4 байта, до 99 КБ — это n до ~800 тысяч).
constexpr int kMaxVerticesForK2 = 32767;
constexpr int kMaxWordsK2 = (kMaxVerticesForK2 + 32) / 32;

template <typename GainT, bool kWide>
__global__ void KernelRebuild2(const uint32_t* __restrict__ bits, int words, int n,
                               const int* __restrict__ degrees, const int* __restrict__ labels, GainT* g,
                               int* sizes, long long* f, long long edges) {
  const int sol = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int* my_labels = labels + (size_t)sol * n;
  GainT* my_g = g + (size_t)sol * 2 * n;

  uint32_t* mask;
  if constexpr (kWide) {
    extern __shared__ unsigned char dynamic_shared[];
    mask = reinterpret_cast<uint32_t*>(dynamic_shared);
  } else {
    __shared__ uint32_t static_mask[kMaxWordsK2];
    mask = static_mask;
  }
  __shared__ long long reduce[kBlockSize];

  int ones = 0;
  for (int w = tid >> 5; w < words; w += kBlockSize / 32) {
    const int v = w * 32 + lane;
    const uint32_t word = __ballot_sync(0xffffffffu, v < n && my_labels[v] != 0);
    if (lane == 0) {
      mask[w] = word;
      ones += __popc(word);
    }
  }
  __syncthreads();

  long long intra_twice = 0;
  for (int v = tid; v < n; v += kBlockSize) {
    const uint32_t* row = bits + (size_t)v * words;
    int g1 = 0;
    for (int w = 0; w < words; ++w) g1 += __popc(row[w] & mask[w]);
    const int g0 = degrees[v] - g1;
    my_g[v] = (GainT)g0;
    my_g[n + v] = (GainT)g1;
    intra_twice += (mask[v >> 5] >> (v & 31)) & 1u ? g1 : g0;
  }

  // Две суммы за одну редукцию: число единиц в старших битах, удвоенные внутренние рёбра в младших.
  reduce[tid] = ((long long)ones << 40) + intra_twice;
  __syncthreads();
  for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) reduce[tid] += reduce[tid + stride];
    __syncthreads();
  }
  if (tid == 0) {
    int my_sizes[2];
    my_sizes[1] = (int)(reduce[0] >> 40);
    my_sizes[0] = n - my_sizes[1];
    sizes[(size_t)sol * 2] = my_sizes[0];
    sizes[(size_t)sol * 2 + 1] = my_sizes[1];
    f[sol] = Objective(edges, my_sizes, 2, (reduce[0] & ((1LL << 40) - 1)) / 2);
  }
}

// Наискорейший спуск по одиночным переносам вершин, до локального оптимума. Один шаблон на два
// варианта хранения состояния решения:
//   kShared = false  G и метки читаются прямо из глобальной памяти;
//   kShared = true   G и метки на время поиска скопированы в разделяемую память блока (нужно
//                    2*k*n + n байт, хост проверяет это заранее). Метки в этом случае —
//                    signed char, а не int: это экономит n*3 байта разделяемой памяти, от чего
//                    напрямую зависит, какие инстансы в неё вообще влезают.
// kShared — компилируемая константа, поэтому if constexpr выбрасывает неиспользуемую ветку ещё
// на этапе компиляции. Оба варианта реализуют один и тот же спуск с одинаковым разрешением ничьих
// и обязаны приходить к одинаковому f — это и проверяется тестами.
//
// На ход приходится один __syncthreads, и держится это на двух вещах:
//   * поток владеет вершинами v = tid (mod kThreads) и в скане, и в правке G, поэтому читает и
//     пишет только свои элементы G и меток, и барьер после применения хода не нужен;
//   * наилучший ход сворачивается __shfl внутри варпа, а итог по варпам каждый поток досчитывает
//     сам. Буфер варпов двойной: запись в него на ходе i + 2 возможна только после барьера хода
//     i + 1, а к нему все потоки приходят, уже дочитав ход i.
// Размеры кластеров лежат в том же двойном режиме. В скане хода i буфер i & 1 хранит размеры до
// хода i - 1, а поправку за сам ход i - 1 каждый поток прибавляет из регистров. После барьера хода i
// этот буфер уже дочитан, и его сразу доводят до размеров после хода i. Держать размеры в массиве
// на поток нельзя: он уходит в локальную память (256 байт стека, проверяйте -Xptxas -v), и при
// k >= 16 и популяции 1024 ядро замедляется на десятки процентов.
// Ход упакован в 64-битный ключ (дельта, вершина, откуда, куда), и минимум ключа даёт то же
// разрешение ничьих, что и CPU: меньшая дельта, затем меньшая вершина, затем меньший кластер.
// GainT — тип элементов G: short на узком пути, int на широком.
template <bool kShared, int kThreads, typename GainT>
__global__ void __launch_bounds__(kThreads)
KernelLocalSearch(const uint32_t* __restrict__ bits, int words, int n, int k, int* labels, GainT* g,
                  int* sizes, long long* f, unsigned long long* accepted_moves) {
  using Label = typename std::conditional<kShared, signed char, int>::type;
  constexpr int kWarps = kThreads / 32;
  static_assert(kThreads % 32 == 0, "block must be whole warps");

  const int sol = blockIdx.x;
  const int tid = threadIdx.x;

  int* my_labels = labels + (size_t)sol * n;
  GainT* my_g = g + (size_t)sol * k * n;
  int* my_sizes = sizes + (size_t)sol * k;

  __shared__ long long warp_best[2][kWarps];

  GainT* wg;  // рабочее G текущего решения
  Label* wl;  // рабочие метки текущего решения
  if constexpr (kShared) {
    extern __shared__ unsigned char dynamic_shared[];
    GainT* sg = reinterpret_cast<GainT*>(dynamic_shared);           // k * n элементов
    Label* slabel = reinterpret_cast<Label*>(sg + (size_t)k * n);  // n элементов
    // Копирование идёт по той же схеме владения, что и спуск, поэтому барьер после него не нужен.
    for (int c = 0; c < k; ++c) {
      for (int v = tid; v < n; v += kThreads) sg[(size_t)c * n + v] = my_g[(size_t)c * n + v];
    }
    for (int v = tid; v < n; v += kThreads) slabel[v] = (Label)my_labels[v];
    wg = sg;
    wl = slabel;
  } else {
    wg = my_g;
    wl = my_labels;
  }
  __shared__ int cluster_size[2][kMaxClusters];
  for (int c = tid; c < k; c += kThreads) cluster_size[0][c] = cluster_size[1][c] = my_sizes[c];
  __syncthreads();
  long long objective = f[sol];

  // "Хода нет": дельта 0 и максимальная нагрузка, больше любого улучшающего ключа.
  constexpr long long kNoMove = 0x7fffffffLL;
  unsigned long long applied = 0;
  int parity = 0;
  int prev_from = -1;  // последний применённый ход, ещё не учтённый в буфере parity
  int prev_to = -1;
  while (true) {
    const int* sizes_now = cluster_size[parity];
    auto size_of = [&](int c) { return sizes_now[c] + (c == prev_to) - (c == prev_from); };

    int local_delta = 0;
    int local_vertex = -1;
    int local_from = 0;
    int local_target = 0;

    for (int v = tid; v < n; v += kThreads) {
      const int from = wl[v];
      const int size_from = size_of(from);
      const int g_from = wg[from * n + v];
      for (int c = 0; c < k; ++c) {
        if (c == from) continue;
        const int delta = MoveDelta(size_from, size_of(c), g_from, wg[c * n + v]);
        if (delta < local_delta) {
          local_delta = delta;
          local_vertex = v;
          local_from = from;
          local_target = c;
        }
      }
    }

    long long key = kNoMove;
    if (local_vertex >= 0) {
      key = (long long)local_delta * 4294967296LL +
            (long long)(((unsigned)local_vertex << 12) | ((unsigned)local_from << 6) | (unsigned)local_target);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      const long long other = __shfl_down_sync(0xffffffffu, key, offset);
      if (other < key) key = other;
    }
    if ((tid & 31) == 0) warp_best[parity][tid >> 5] = key;
    __syncthreads();

    long long best = warp_best[parity][0];
    for (int w = 1; w < kWarps; ++w) {
      const long long other = warp_best[parity][w];
      if (other < best) best = other;
    }
    if (best >= 0) break;  // улучшающих ходов не осталось

    const int delta = (int)(best >> 32);
    const unsigned payload = (unsigned)(best & 0xffffffffLL);
    const int v = (int)(payload >> 12);
    const int from = (int)((payload >> 6) & 63u);
    const int to = (int)(payload & 63u);

    // Буфер parity дочитан всеми (они прошли барьер): доводим его до размеров после этого хода.
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
      if ((row[u >> 5] >> (u & 31)) & 1u) {
        g_from[u] -= 1;
        g_to[u] += 1;
      }
    }
    if (v % kThreads == tid) wl[v] = (Label)to;
    objective += delta;
    ++applied;
  }

  if constexpr (kShared) {
    for (int c = 0; c < k; ++c) {
      for (int v = tid; v < n; v += kThreads) my_g[(size_t)c * n + v] = wg[(size_t)c * n + v];
    }
    for (int v = tid; v < n; v += kThreads) my_labels[v] = wl[v];
  }
  // После break буфер parity никто больше не пишет: к нему остаётся прибавить последний ход.
  for (int c = tid; c < k; c += kThreads) {
    my_sizes[c] = cluster_size[parity][c] + (c == prev_to) - (c == prev_from);
  }
  if (tid == 0) {
    f[sol] = objective;
    if (accepted_moves != nullptr) atomicAdd(accepted_moves, applied);
  }
}

// Тот же спуск для k = 2, где у вершины единственный ход — в другой кластер. Состояние вершины
// сводится к одному числу e = G[1][v] - G[0][v], и оно не зависит от её метки:
//   перенос 0 -> 1 стоит Δf = s - 2e, перенос 1 -> 0 стоит Δf = 2 - s + 2e, где s = n1 - n0 + 1;
//   перенос любой вершины v в кластер to меняет e у всех соседей v на одно и то же число
//   (+2 при to = 1, -2 при to = 0), а e самой v не меняет.
// Поэтому всё состояние особи помещается в регистры потока: e и бит метки на каждую из kVpt
// вершин, которыми поток владеет (u = tid + j*kThreads), и n0, который каждый поток ведёт сам,
// раз все видят выбранный ход. Память за ход читает только строку смежности v; вершины одного
// варпа лежат в одном её слове, так что загрузка одна на варп, а бит потока — всегда бит lane.
// Правка и скан следующего хода слиты в один проход; барьер на ход один, буфер варпов двойной, по
// той же схеме, что и у общего ядра. Наилучший ход потока ищется строгим сравнением по возрастанию
// j, то есть с меньшей вершиной при равной дельте, а ключ собирается один раз на поток:
// (Δ + 32768) << 16 | u << 1 | метка. Перенос вершины меняет только n - 1 пар с её участием, так что
// улучшающая Δ лежит в [1 - n, -1] и при n <= kMaxVerticesForK2 укладывается в 15 бит, а минимум
// ключа даёт то же разрешение ничьих, что и CPU.
constexpr unsigned kNoMove2 = 0xffffffffu;

__device__ __forceinline__ unsigned WarpMin(unsigned key) {
#if __CUDA_ARCH__ >= 800
  return __reduce_min_sync(0xffffffffu, key);
#else
  for (int offset = 16; offset > 0; offset >>= 1) key = min(key, __shfl_xor_sync(0xffffffffu, key, offset));
  return key;
#endif
}

template <int kThreads, int kVpt>
__global__ void __launch_bounds__(kThreads)
KernelLocalSearch2(const uint32_t* __restrict__ bits, int words, int n, int* labels, Gain* g, int* sizes,
                   long long* f, unsigned long long* accepted_moves) {
  constexpr int kWarps = kThreads / 32;
  static_assert(kThreads % 32 == 0 && kWarps <= 32, "block must be 1..32 whole warps");
  static_assert(kVpt <= 32, "labels of a thread are packed into 32 bits");

  const int sol = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const uint32_t lane_bit = 1u << lane;
  int* my_labels = labels + (size_t)sol * n;
  Gain* my_g = g + (size_t)sol * 2 * n;

  __shared__ unsigned warp_best[2][kWarps];

  // Вершины за пределами n получают e, при котором их перенос никогда не улучшает f, и не
  // меняют его: правку отсекает маска active.
  constexpr int kIdle = -(1 << 29);
  int e[kVpt];
  unsigned lab = 0;
  unsigned active = 0;
#pragma unroll
  for (int j = 0; j < kVpt; ++j) {
    const int v = tid + j * kThreads;
    e[j] = kIdle;
    if (v < n) {
      e[j] = (int)my_g[n + v] - (int)my_g[v];
      lab |= (unsigned)my_labels[v] << j;
      active |= 1u << j;
    }
  }
  int n0 = sizes[(size_t)sol * 2];
  long long objective = f[sol];

  // Ключ наилучшего улучшающего хода потока.
  auto best_key = [&](int best_delta, int best_j) {
    if (best_delta >= 0) return kNoMove2;
    const unsigned u = (unsigned)(tid + best_j * kThreads);
    return ((unsigned)(best_delta + 32768) << 16) | (u << 1) | ((lab >> best_j) & 1u);
  };

  int best_delta = 0;
  int best_j = 0;
  {
    const int s = n - 2 * n0 + 1;
#pragma unroll
    for (int j = 0; j < kVpt; ++j) {
      const int delta = ((lab >> j) & 1u) ? 2 - s + 2 * e[j] : s - 2 * e[j];
      if (delta < best_delta) {
        best_delta = delta;
        best_j = j;
      }
    }
  }
  unsigned key = best_key(best_delta, best_j);

  unsigned long long applied = 0;
  int parity = 0;
  while (true) {
    key = WarpMin(key);
    if (lane == 0) warp_best[parity][tid >> 5] = key;
    __syncthreads();
    const unsigned best = WarpMin(warp_best[parity][lane < kWarps ? lane : 0]);
    if (best == kNoMove2) break;
    parity ^= 1;

    const int delta = (int)(best >> 16) - 32768;
    const int v = (int)((best >> 1) & 0x7fffu);
    const unsigned from = best & 1u;
    n0 += from ? 1 : -1;
    const int s = n - 2 * n0 + 1;
    const int step = from ? -2 : 2;
    if (v % kThreads == tid) lab ^= 1u << (v / kThreads);

    const uint32_t* row = bits + (size_t)v * words;
    best_delta = 0;
    best_j = 0;
#pragma unroll
    for (int j = 0; j < kVpt; ++j) {
      // За конец строки могут выйти только вершины за пределами n: им хватит последнего слова.
      const int word = min((tid + j * kThreads) >> 5, words - 1);
      if ((__ldg(row + word) & lane_bit) && ((active >> j) & 1u)) e[j] += step;
      const int du = ((lab >> j) & 1u) ? 2 - s + 2 * e[j] : s - 2 * e[j];
      if (du < best_delta) {
        best_delta = du;
        best_j = j;
      }
    }
    key = best_key(best_delta, best_j);
    objective += delta;
    ++applied;
  }

  // G восстанавливается из e и степени вершины: глобальное G за время спуска не менялось, а сумма
  // его строк по кластерам и есть степень.
#pragma unroll
  for (int j = 0; j < kVpt; ++j) {
    const int v = tid + j * kThreads;
    if (v < n) {
      const int degree = (int)my_g[v] + (int)my_g[n + v];
      my_labels[v] = (lab >> j) & 1u;
      my_g[v] = (Gain)((degree - e[j]) / 2);
      my_g[n + v] = (Gain)((degree + e[j]) / 2);
    }
  }
  if (tid == 0) {
    sizes[(size_t)sol * 2] = n0;
    sizes[(size_t)sol * 2 + 1] = n - n0;
    f[sol] = objective;
    if (accepted_moves != nullptr) atomicAdd(accepted_moves, applied);
  }
}

// Минимум 64-битных ключей по варпу. На sm_80+ это две 32-битные редукции: сначала старшая половина,
// затем младшая среди потоков, у которых старшая совпала с минимумом.
__device__ __forceinline__ unsigned long long WarpMin64(unsigned long long key) {
#if __CUDA_ARCH__ >= 800
  const unsigned high = __reduce_min_sync(0xffffffffu, (unsigned)(key >> 32));
  const unsigned low = __reduce_min_sync(0xffffffffu, (unsigned)(key >> 32) == high ? (unsigned)key : 0xffffffffu);
  return ((unsigned long long)high << 32) | low;
#else
  for (int offset = 16; offset > 0; offset >>= 1) {
    const unsigned long long other = __shfl_xor_sync(0xffffffffu, key, offset);
    if (other < key) key = other;
  }
  return key;
#endif
}

// Спуск k = 2 для широкого пути, где e и метки уже не помещаются в регистры блока (n > 32 768).
// Алгебра та же, что у KernelLocalSearch2, но состояние вершины лежит в глобальной памяти одним словом
// w = 2e + метка — на месте строки G[0] особи: во время спуска G не нужна, а в конце восстанавливается из
// e и степени. Через w дельта считается напрямую: s - w при метке 0 и w + 1 - s при метке 1. За ход поток
// читает слово каждой своей вершины и строку смежности v (одна загрузка на варп, бит потока — бит lane),
// пишет — только соседям v. Правка слита со сканом следующего хода, барьер на ход один. Ключ 64-битный:
// (Δ + n) << 32 | u << 1 | метка; улучшающая Δ лежит в [1 - n, -1], так что старшая половина
// положительна, а порядок ключей даёт то же разрешение ничьих, что и CPU.
//
// Сетка постоянная: блоков запускается столько, чтобы их состояние (4n байт на особь) помещалось в L2, и
// блок берёт следующую особь из счётчика next_individual, пока они не кончатся. Когда состояние всех
// работающих блоков в L2 не помещается, ход упирается в пропускную способность видеопамяти.
constexpr unsigned long long kNoMoveWide = ~0ull;

template <int kThreads>
__global__ void __launch_bounds__(kThreads)
KernelLocalSearch2Wide(const uint32_t* __restrict__ bits, int words, int n, int population,
                       const int* __restrict__ degrees, int* labels, WideGain* g, int* sizes, long long* f,
                       unsigned long long* accepted_moves, int* next_individual) {
  constexpr int kWarps = kThreads / 32;
  static_assert(kThreads % 32 == 0 && kWarps <= 32, "block must be 1..32 whole warps");

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const uint32_t lane_bit = 1u << lane;

  __shared__ unsigned long long warp_best[2][kWarps];
  __shared__ int current;

  while (true) {
    if (tid == 0) current = atomicAdd(next_individual, 1);
    __syncthreads();
    const int sol = current;
    if (sol >= population) break;

    int* my_labels = labels + (size_t)sol * n;
    WideGain* my_g = g + (size_t)sol * 2 * n;
    int* state = my_g;  // строка G[0] на время спуска

    auto key_of = [&](int best_delta, int best_u, int best_word) {
      if (best_u < 0) return kNoMoveWide;
      return ((unsigned long long)(best_delta + n) << 32) | ((unsigned)best_u << 1) | (unsigned)(best_word & 1);
    };

    int n0 = sizes[(size_t)sol * 2];
    long long objective = f[sol];
    int s = n - 2 * n0 + 1;
    int best_delta = 0;
    int best_u = -1;
    int best_word = 0;
    for (int u = tid; u < n; u += kThreads) {
      const int word = 2 * (my_g[n + u] - my_g[u]) + my_labels[u];
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
      key = WarpMin64(key);
      if (lane == 0) warp_best[parity][tid >> 5] = key;
      __syncthreads();
      const unsigned long long best = WarpMin64(warp_best[parity][lane < kWarps ? lane : 0]);
      if (best == kNoMoveWide) break;
      parity ^= 1;

      const int delta = (int)(best >> 32) - n;
      const int v = (int)((unsigned)best >> 1);
      const int from = (int)(best & 1u);
      n0 += from ? 1 : -1;
      s = n - 2 * n0 + 1;
      const int shift = from ? -4 : 4;  // e соседа меняется на -+2, слово — вдвое больше
      if (v % kThreads == tid) state[v] ^= 1;

      const uint32_t* row = bits + (size_t)v * words;
      best_delta = 0;
      best_u = -1;
#pragma unroll 4
      for (int u = tid; u < n; u += kThreads) {
        int word = state[u];
        if (__ldg(row + (u >> 5)) & lane_bit) {
          word += shift;
          state[u] = word;
        }
        const int du = (word & 1) ? word + 1 - s : s - word;
        if (du < best_delta) {
          best_delta = du;
          best_u = u;
          best_word = word;
        }
      }
      key = key_of(best_delta, best_u, best_word);
      objective += delta;
      ++applied;
    }

    for (int u = tid; u < n; u += kThreads) {
      const int word = state[u];
      const int e = word >> 1;
      const int degree = degrees[u];
      my_labels[u] = word & 1;
      my_g[u] = (degree - e) / 2;
      my_g[n + u] = (degree + e) / 2;
    }
    if (tid == 0) {
      sizes[(size_t)sol * 2] = n0;
      sizes[(size_t)sol * 2 + 1] = n - n0;
      f[sol] = objective;
      if (accepted_moves != nullptr) atomicAdd(accepted_moves, applied);
    }
    // warp_best и current перепишет уже следующая особь: все должны дочитать их для этой.
    __syncthreads();
  }
}

// Спуск k = 3 для широкого пути. Состояние вершины — одно 64-битное слово: три счётчика G по 20 бит
// (n < 2^20, так что G <= n - 1 в поле помещается) и метка в двух старших битах. Слова лежат в отдельном
// буфере work (8n байт на особь, вдвое меньше, чем G и метки общего ядра), G и метки восстанавливаются в
// конце. Перенос соседа v из from в to правит слово одним сложением: G[from] >= 1 (там сам v), а G[to] + 1 не
// больше степени, так что переносов между полями не бывает. Дельта хода — через h_c = s_c - 2 G[c]:
// Δ(a -> c) = h_c - h_a + 1, и лучшая цель вершины — меньший h из двух чужих кластеров, при равенстве меньший
// номер. Размеры кластеров каждый поток ведёт в регистрах: выбранный ход видят все. Ключ
// (Δ + n) << 24 | u << 4 | откуда << 2 | куда даёт то же разрешение ничьих, что и CPU. Сетка постоянная, как
// у KernelLocalSearch2Wide.
constexpr int kFieldBits = 20;
constexpr unsigned long long kFieldMask = (1ull << kFieldBits) - 1;
static_assert(kMaxVerticesWide <= (1 << kFieldBits), "G must fit in a 20-bit field of the packed k = 3 word");

template <int kThreads>
__global__ void __launch_bounds__(kThreads)
KernelLocalSearch3Wide(const uint32_t* __restrict__ bits, int words, int n, int population, int* labels,
                       WideGain* g, int* sizes, long long* f, unsigned long long* work,
                       unsigned long long* accepted_moves, int* next_individual) {
  constexpr int kWarps = kThreads / 32;
  static_assert(kThreads % 32 == 0 && kWarps <= 32, "block must be 1..32 whole warps");

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const uint32_t lane_bit = 1u << lane;

  __shared__ unsigned long long warp_best[2][kWarps];
  __shared__ int current;

  while (true) {
    if (tid == 0) current = atomicAdd(next_individual, 1);
    __syncthreads();
    const int sol = current;
    if (sol >= population) break;

    int* my_labels = labels + (size_t)sol * n;
    WideGain* my_g = g + (size_t)sol * 3 * n;
    unsigned long long* state = work + (size_t)sol * n;
    int s0 = sizes[(size_t)sol * 3];
    int s1 = sizes[(size_t)sol * 3 + 1];
    int s2 = sizes[(size_t)sol * 3 + 2];
    long long objective = f[sol];

    int best_delta = 0;
    int best_move = -1;  // u << 4 | откуда << 2 | куда
    auto consider = [&](unsigned long long word, int u) {
      const int a = (int)(word >> 62);
      const int h0 = s0 - 2 * (int)(word & kFieldMask);
      const int h1 = s1 - 2 * (int)((word >> kFieldBits) & kFieldMask);
      const int h2 = s2 - 2 * (int)((word >> (2 * kFieldBits)) & kFieldMask);
      const int ha = a == 0 ? h0 : (a == 1 ? h1 : h2);
      const int low = a == 0 ? h1 : h0;  // чужой кластер с меньшим номером
      const int high = a == 2 ? h1 : h2;
      const bool take_high = high < low;
      const int delta = (take_high ? high : low) - ha + 1;
      if (delta < best_delta) {
        best_delta = delta;
        best_move = (u << 4) | (a << 2) | (take_high ? (a == 2 ? 1 : 2) : (a == 0 ? 1 : 0));
      }
    };
    auto key_of = [&]() {
      if (best_move < 0) return kNoMoveWide;
      return ((unsigned long long)(best_delta + n) << 24) | (unsigned)best_move;
    };

    for (int u = tid; u < n; u += kThreads) {
      const unsigned long long word = (unsigned long long)my_g[u] |
                                      ((unsigned long long)my_g[n + u] << kFieldBits) |
                                      ((unsigned long long)my_g[2 * n + u] << (2 * kFieldBits)) |
                                      ((unsigned long long)my_labels[u] << 62);
      state[u] = word;
      consider(word, u);
    }
    unsigned long long key = key_of();

    unsigned long long applied = 0;
    int parity = 0;
    while (true) {
      key = WarpMin64(key);
      if (lane == 0) warp_best[parity][tid >> 5] = key;
      __syncthreads();
      const unsigned long long best = WarpMin64(warp_best[parity][lane < kWarps ? lane : 0]);
      if (best == kNoMoveWide) break;
      parity ^= 1;

      const int delta = (int)(best >> 24) - n;
      const int move = (int)(best & 0xffffffu);
      const int v = move >> 4;
      const int from = (move >> 2) & 3;
      const int to = move & 3;
      s0 += (to == 0) - (from == 0);
      s1 += (to == 1) - (from == 1);
      s2 += (to == 2) - (from == 2);
      const unsigned long long inc = (1ull << (kFieldBits * to)) - (1ull << (kFieldBits * from));
      if (v % kThreads == tid) state[v] = (state[v] & ~(3ull << 62)) | ((unsigned long long)to << 62);

      const uint32_t* row = bits + (size_t)v * words;
      best_delta = 0;
      best_move = -1;
#pragma unroll 4
      for (int u = tid; u < n; u += kThreads) {
        unsigned long long word = state[u];
        if (__ldg(row + (u >> 5)) & lane_bit) {
          word += inc;
          state[u] = word;
        }
        consider(word, u);
      }
      key = key_of();
      objective += delta;
      ++applied;
    }

    for (int u = tid; u < n; u += kThreads) {
      const unsigned long long word = state[u];
      my_g[u] = (WideGain)(word & kFieldMask);
      my_g[n + u] = (WideGain)((word >> kFieldBits) & kFieldMask);
      my_g[2 * n + u] = (WideGain)((word >> (2 * kFieldBits)) & kFieldMask);
      my_labels[u] = (int)(word >> 62);
    }
    if (tid == 0) {
      sizes[(size_t)sol * 3] = s0;
      sizes[(size_t)sol * 3 + 1] = s1;
      sizes[(size_t)sol * 3 + 2] = s2;
      f[sol] = objective;
      if (accepted_moves != nullptr) atomicAdd(accepted_moves, applied);
    }
    __syncthreads();  // warp_best и current перепишет уже следующая особь
  }
}

// Потоков на блок локального поиска. Правило снято замером на RTX 4070 Ti SUPER (66 SM), сетка
// n = 500..16000, k = 3 и 8, популяция 64..4096. Пока блоков мало, карта недогружена, и больший блок
// быстрее — до 1,95x при популяции 64. Когда популяция уже заполняет SM, а работы на особь мало
// (k*n < 10000), выгоднее 256: у 512 потоков остаётся по несколько вершин на поток, и ход съедает
// синхронизация — там 256 быстрее до 1,6x. В остальных точках разница в пределах нескольких процентов.
int LocalSearchThreads(int n, int k, int population) {
  return (population >= 256 && (long long)k * n < 10000) ? 256 : 512;
}

template <int kThreads>
void LaunchLocalSearch2(int vpt, int population, const uint32_t* bits, int words, int n, int* labels, Gain* g,
                        int* sizes, long long* f, unsigned long long* moves) {
  auto launch = [&](auto vpt_constant) {
    constexpr int kVpt = decltype(vpt_constant)::value;
    KernelLocalSearch2<kThreads, kVpt><<<population, kThreads>>>(bits, words, n, labels, g, sizes, f, moves);
  };
  switch (vpt) {
    case 1: return launch(std::integral_constant<int, 1>{});
    case 2: return launch(std::integral_constant<int, 2>{});
    case 4: return launch(std::integral_constant<int, 4>{});
    case 8: return launch(std::integral_constant<int, 8>{});
    case 12: return launch(std::integral_constant<int, 12>{});
    case 16: return launch(std::integral_constant<int, 16>{});
    case 24: return launch(std::integral_constant<int, 24>{});
    case 32: return launch(std::integral_constant<int, 32>{});
  }
  throw std::runtime_error("unsupported vertices per thread: " + std::to_string(vpt));
}

// Наименьшее поддержанное число вершин на поток, при котором блок из threads потоков покрывает n.
int VerticesPerThread2(int n, int threads) {
  const int need = (n + threads - 1) / threads;
  for (int v : {1, 2, 4, 8, 12, 16, 24, 32}) {
    if (v >= need) return v;
  }
  return -1;
}

// Потоков на блок ядра k = 2: наименьший блок, в котором на поток приходится не больше 12 вершин.
// Снято замером на RTX 4070 Ti SUPER, n = 300..14000, популяция 64..1024. Узкое место — число
// инструкций на вершину за ход, и лишние потоки только удлиняют редукцию; но 16 вершин на поток уже
// упираются в регистры и бывают в 2-4 раза медленнее 8. Исключение — малая популяция на графе до
// 1024 вершин: карта там недогружена, и блок 256 быстрее блока 128 до 1,7x.
int LocalSearchThreads2(int n, int population) {
  if (population < 256 && n > 512 && n <= 1024) return 256;
  for (int threads : {128, 256, 512}) {
    if (n <= 12 * threads) return threads;
  }
  return 1024;
}

void DispatchLocalSearch2(int population, const uint32_t* bits, int words, int n, int* labels, Gain* g,
                          int* sizes, long long* f, unsigned long long* moves) {
  const int threads = LocalSearchThreads2(n, population);
  const int vpt = VerticesPerThread2(n, threads);
  switch (threads) {
    case 128: LaunchLocalSearch2<128>(vpt, population, bits, words, n, labels, g, sizes, f, moves); return;
    case 256: LaunchLocalSearch2<256>(vpt, population, bits, words, n, labels, g, sizes, f, moves); return;
    case 512: LaunchLocalSearch2<512>(vpt, population, bits, words, n, labels, g, sizes, f, moves); return;
    default: LaunchLocalSearch2<1024>(vpt, population, bits, words, n, labels, g, sizes, f, moves); return;
  }
}

template <bool kShared, typename GainT>
void LaunchLocalSearch(int threads, int population, size_t shared_bytes, const uint32_t* bits, int words,
                       int n, int k, int* labels, GainT* g, int* sizes, long long* f,
                       unsigned long long* moves) {
  if constexpr (!kShared && std::is_same<GainT, WideGain>::value) {
    if (threads == 1024) {
      KernelLocalSearch<false, 1024, WideGain><<<population, 1024>>>(bits, words, n, k, labels, g, sizes, f, moves);
      return;
    }
  }
  if (threads == 256) {
    KernelLocalSearch<kShared, 256, GainT><<<population, 256, shared_bytes>>>(bits, words, n, k, labels, g, sizes,
                                                                              f, moves);
  } else {
    KernelLocalSearch<kShared, 512, GainT><<<population, 512, shared_bytes>>>(bits, words, n, k, labels, g, sizes,
                                                                              f, moves);
  }
}

// Широкий путь всегда идёт блоками по 1024 потока: на вершину там приходятся сотни итераций цикла, и
// лишние потоки только помогают. Общее ядро (k >= 4) с 1024 потоками быстрее, чем с 512, в 1,1–1,3 раза
// (выборка Stack Overflow в 60 000 вершин, k = 4 и 6). Ядра k = 2 и k = 3 идут постоянной сеткой, в которой
// блоков меньше, чем SM, и каждому блоку лучше занять свой SM целиком.
constexpr int kWideThreads = 1024;

// Доля L2 под состояние особей, которые спускаются одновременно. Меньше блоков — простаивают SM, больше —
// состояние вытесняется из L2 в видеопамять; в обе стороны итерация дорожает на 10–25 %. Снято замером на
// RTX 4070 Ti SUPER (48 МБ L2) на всём NUS-WIDE (193 734 вершины) и выборке Stack Overflow в 330 000 вершин:
// лучшее число блоков занимало при k = 2 52 и 74 % L2, при k = 3 от 72 до 110 % (там разница в пределах 3 %).
constexpr double kL2ShareK2 = 0.6;
constexpr double kL2ShareK3 = 0.8;

// Запуски, которые на двух путях идут разными ядрами, разведены перегрузками по типу G, а не if constexpr:
// nvcc не всегда отбрасывает ветку if constexpr с запуском ядра внутри шаблона.
void LaunchRebuild2(int population, size_t mask_bytes, const uint32_t* bits, int words, int n, const int* degrees,
                    const int* labels, Gain* g, int* sizes, long long* f, long long edges) {
  (void)mask_bytes;
  KernelRebuild2<Gain, false><<<population, kBlockSize>>>(bits, words, n, degrees, labels, g, sizes, f, edges);
}

void LaunchRebuild2(int population, size_t mask_bytes, const uint32_t* bits, int words, int n, const int* degrees,
                    const int* labels, WideGain* g, int* sizes, long long* f, long long edges) {
  KernelRebuild2<WideGain, true><<<population, kBlockSize, mask_bytes>>>(bits, words, n, degrees, labels, g, sizes,
                                                                         f, edges);
}

void DispatchLocalSearch2(int blocks, int population, const uint32_t* bits, int words, int n, const int* degrees,
                          int* labels, Gain* g, int* sizes, long long* f, unsigned long long* moves,
                          int* next_individual) {
  (void)blocks;
  (void)degrees;
  (void)next_individual;
  DispatchLocalSearch2(population, bits, words, n, labels, g, sizes, f, moves);
}

void DispatchLocalSearch2(int blocks, int population, const uint32_t* bits, int words, int n, const int* degrees,
                          int* labels, WideGain* g, int* sizes, long long* f, unsigned long long* moves,
                          int* next_individual) {
  CC_CUDA_CHECK(cudaMemsetAsync(next_individual, 0, sizeof(int)));
  KernelLocalSearch2Wide<kWideThreads><<<blocks, kWideThreads>>>(bits, words, n, population, degrees, labels, g,
                                                                 sizes, f, moves, next_individual);
}

void LaunchLocalSearch3(int blocks, int population, const uint32_t* bits, int words, int n, int* labels, Gain* g,
                        int* sizes, long long* f, unsigned long long* work, unsigned long long* moves,
                        int* next_individual) {
  // На узком пути k = 3 идёт общим ядром; перегрузка нужна только для единообразного вызова из Solve.
  (void)blocks, (void)population, (void)bits, (void)words, (void)n, (void)labels, (void)g, (void)sizes;
  (void)f, (void)work, (void)moves, (void)next_individual;
  throw std::logic_error("k = 3 wide kernel called on the narrow path");
}

void LaunchLocalSearch3(int blocks, int population, const uint32_t* bits, int words, int n, int* labels,
                        WideGain* g, int* sizes, long long* f, unsigned long long* work, unsigned long long* moves,
                        int* next_individual) {
  CC_CUDA_CHECK(cudaMemsetAsync(next_individual, 0, sizeof(int)));
  KernelLocalSearch3Wide<kWideThreads><<<blocks, kWideThreads>>>(bits, words, n, population, labels, g, sizes, f,
                                                                 work, moves, next_individual);
}

// Сколько блоков постоянной сетки запускать: не больше, чем помещается на карту, и столько, чтобы состояние
// работающих особей (state_bytes на каждую) занимало не больше доли l2_share кэша L2.
int ResidentBlocks(const void* kernel, size_t state_bytes, int population, double l2_share) {
  int device = 0;
  CC_CUDA_CHECK(cudaGetDevice(&device));
  int sms = 0;
  int l2_bytes = 0;
  CC_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  CC_CUDA_CHECK(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, device));
  int per_sm = 0;
  CC_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kWideThreads, 0));
  const long long fit = (long long)(l2_share * l2_bytes / (double)std::max<size_t>(state_bytes, 1));
  const long long blocks = std::min<long long>(std::max(1, sms * per_sm), std::max(1LL, fit));
  return (int)std::min<long long>(blocks, population);
}

// Турнирная селекция. Один блок на слот следующей популяции; метки, G и размеры победителя
// копируются целиком, поэтому пересчёт после неё не нужен.
template <typename GainT>
__global__ void KernelSelect(int n, int k, int population, int tournament,
                             const int* __restrict__ labels, const GainT* __restrict__ g,
                             const int* __restrict__ sizes, const long long* __restrict__ f,
                             int* out_labels, GainT* out_g, int* out_sizes, long long* out_f,
                             uint64_t seed, int iteration) {
  const int slot = blockIdx.x;
  const int tid = threadIdx.x;

  __shared__ int winner;
  if (tid == 0) {
    uint64_t rng = StreamSeed(seed, slot, iteration + 1);
    int chosen = RandBelow(rng, population);
    for (int t = 1; t < tournament; ++t) {
      const int candidate = RandBelow(rng, population);
      if (f[candidate] < f[chosen]) chosen = candidate;
    }
    winner = chosen;
    out_f[slot] = f[chosen];
  }
  __syncthreads();

  // Индексы считаются отдельно в каждом цикле (а не через общие указатели
  // "источник"/"приёмник"), чтобы регистры под адрес одного массива
  // освобождались до начала следующего цикла — иначе компилятор держит все
  // шесть указателей живыми одновременно, и копирующее ядро занимает на SM
  // меньше блоков.
  const int source = winner;
  for (int v = tid; v < n; v += kBlockSize) {
    out_labels[(size_t)slot * n + v] = labels[(size_t)source * n + v];
  }
  for (int i = tid; i < k * n; i += kBlockSize) {
    out_g[(size_t)slot * k * n + i] = g[(size_t)source * k * n + i];
  }
  for (int c = tid; c < k; c += kBlockSize) {
    out_sizes[(size_t)slot * k + c] = sizes[(size_t)source * k + c];
  }
}

// Перемаркирует каждую вершину с заданной вероятностью; G пересчитывается после.
template <typename Index>
__global__ void KernelPerturb(Index total, int k, float probability, int* labels, uint64_t seed,
                              int iteration) {
  const Index stride = blockDim.x * gridDim.x;
  for (Index idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
    uint64_t rng = StreamSeed(seed, idx, iteration + 0x5000000ull);
    if (RandFloat(rng) >= probability) continue;
    const int shift = 1 + (int)RandBelow(rng, k - 1);
    labels[idx] = (labels[idx] + shift) % k;
  }
}

// Буферы устройства для одного запуска.
template <typename GainT>
struct DeviceArena {
  uint32_t* bits = nullptr;
  int* labels = nullptr;
  int* labels_next = nullptr;
  GainT* g = nullptr;
  GainT* g_next = nullptr;
  int* sizes = nullptr;
  int* sizes_next = nullptr;
  long long* f = nullptr;
  long long* f_next = nullptr;
  uint32_t* masks = nullptr;
  unsigned long long* moves = nullptr;
  int* degrees = nullptr;
  int* next_individual = nullptr;
  unsigned long long* work = nullptr;

  ~DeviceArena() {
    cudaFree(bits);
    cudaFree(labels);
    cudaFree(labels_next);
    cudaFree(g);
    cudaFree(g_next);
    cudaFree(sizes);
    cudaFree(sizes_next);
    cudaFree(f);
    cudaFree(f_next);
    cudaFree(masks);
    cudaFree(moves);
    cudaFree(degrees);
    cudaFree(next_individual);
    cudaFree(work);
  }
};

std::string Mebibytes(size_t bytes) { return std::to_string((bytes + (1u << 20) - 1) >> 20) + " MiB"; }

// Весь запуск на устройстве. GainT выбирает путь: Gain — узкий, как был, WideGain — широкий.
template <typename GainT>
PbilsResult Solve(const Graph& graph, const PbilsParams& params) {
  constexpr bool kWide = std::is_same<GainT, WideGain>::value;
  using Index = typename std::conditional<kWide, long long, int>::type;

  const auto started = std::chrono::steady_clock::now();

  const int n = (int)graph.Size();
  const int k = params.k;
  const int population = params.population;
  const int words = (int)graph.WordsPerRow();
  const long long edges = (long long)graph.EdgeCount();

  // Выбор, где состояние решения живёт во время локального поиска.
  const size_t shared_bytes = sizeof(GainT) * (size_t)k * n + (size_t)n;
  int device = 0;
  CC_CUDA_CHECK(cudaGetDevice(&device));
  int shared_limit = 0;
  CC_CUDA_CHECK(cudaDeviceGetAttribute(&shared_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

  bool use_shared = false;
  if (params.ls_kernel == PbilsParams::kLocalSearchShared) {
    if (kWide || shared_bytes > (size_t)shared_limit) {
      throw std::runtime_error("state needs " + std::to_string(shared_bytes) +
                               " bytes of shared memory, device allows " + std::to_string(shared_limit));
    }
    use_shared = true;
  } else if (params.ls_kernel == PbilsParams::kLocalSearchAuto) {
    use_shared = !kWide && shared_bytes <= (size_t)shared_limit;
  }

  if constexpr (!kWide) {
    if (use_shared) {
      CC_CUDA_CHECK(cudaFuncSetAttribute(KernelLocalSearch<true, 256, Gain>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared_bytes));
      CC_CUDA_CHECK(cudaFuncSetAttribute(KernelLocalSearch<true, 512, Gain>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared_bytes));
    }
  }
  const int ls_threads = kWide ? kWideThreads : LocalSearchThreads(n, k, population);
  // При k = 2 спуск идёт отдельным ядром: на узком пути с состоянием в регистрах, на широком — в глобальной
  // памяти. Явный выбор shared/global оставляет общее ядро, чтобы его можно было сравнить с этим. На широком
  // пути так же идёт и k = 3.
  const bool use_k2 = k == 2 && (kWide ? params.ls_kernel != PbilsParams::kLocalSearchGlobal
                                       : params.ls_kernel == PbilsParams::kLocalSearchAuto && n <= kMaxVerticesForK2);
  const bool use_k3 = kWide && k == 3 && params.ls_kernel != PbilsParams::kLocalSearchGlobal;
  const int ls_threads2 = kWide ? kWideThreads : LocalSearchThreads2(n, population);
  // Блоков постоянной сетки у ядер k = 2 и k = 3 широкого пути; у остальных ядер блок на особь.
  int ls_blocks = population;
  if (kWide && use_k2) {
    ls_blocks = ResidentBlocks((const void*)KernelLocalSearch2Wide<kWideThreads>, (size_t)n * sizeof(int), population,
                               kL2ShareK2);
  } else if (use_k3) {
    ls_blocks = ResidentBlocks((const void*)KernelLocalSearch3Wide<kWideThreads>,
                               (size_t)n * sizeof(unsigned long long), population, kL2ShareK3);
  }
  // Маска кластера 1 для KernelRebuild2 широкого пути лежит в динамической разделяемой памяти.
  const size_t mask_bytes = (size_t)words * sizeof(uint32_t);
  if (kWide && use_k2) {
    if (mask_bytes > (size_t)shared_limit) {
      throw std::runtime_error("cluster mask needs " + std::to_string(mask_bytes) +
                               " bytes of shared memory, device allows " + std::to_string(shared_limit));
    }
    CC_CUDA_CHECK(cudaFuncSetAttribute(KernelRebuild2<WideGain, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       (int)mask_bytes));
  }
  if (params.verbose) {
    if (kWide) {
      std::printf("  local search: wide path (32-bit G), %s kernel, %d threads per block, %d blocks\n",
                  use_k2 ? "k = 2" : (use_k3 ? "k = 3" : "generic"), kWideThreads, ls_blocks);
    } else {
      if (use_k2) {
        std::printf("  local search: k = 2 register kernel, %d threads x %d vertices\n", ls_threads2,
                    VerticesPerThread2(n, ls_threads2));
      }
      std::printf("  local search state: %s (%zu bytes per block, device allows %d), %d threads per block\n",
                  use_shared ? "shared memory" : "global memory", shared_bytes, shared_limit, ls_threads);
    }
  }

  const size_t bits_bytes = graph.Bits().size() * sizeof(uint32_t);
  const size_t labels_count = (size_t)population * n;
  const size_t g_count = (size_t)population * k * n;
  const size_t sizes_count = (size_t)population * k;
  const size_t masks_count = use_k2 ? 0 : (size_t)population * k * words;

  // Под WDDM аллокация сверх физической памяти карты проходит: драйвер молча выносит её в ОЗУ, и ядра
  // замедляются в разы. Поэтому объём сверяется со свободной памятью заранее.
  const size_t device_bytes = bits_bytes + 2 * labels_count * sizeof(int) + 2 * g_count * sizeof(GainT) +
                              2 * sizes_count * sizeof(int) + 2 * (size_t)population * sizeof(long long) +
                              masks_count * sizeof(uint32_t) + (use_k2 ? (size_t)n * sizeof(int) : 0) +
                              (use_k3 ? labels_count * sizeof(unsigned long long) : 0) +
                              sizeof(unsigned long long) + sizeof(int);
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  CC_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  if (device_bytes > free_bytes) {
    throw std::runtime_error("run needs " + Mebibytes(device_bytes) + " of device memory, " + Mebibytes(free_bytes) +
                             " free");
  }
  if (params.verbose) {
    std::printf("  device memory: %s of %s free\n", Mebibytes(device_bytes).c_str(), Mebibytes(free_bytes).c_str());
  }

  DeviceArena<GainT> arena;
  CC_CUDA_CHECK(cudaMalloc(&arena.bits, bits_bytes));
  CC_CUDA_CHECK(cudaMemcpy(arena.bits, graph.Bits().data(), bits_bytes, cudaMemcpyHostToDevice));
  CC_CUDA_CHECK(cudaMalloc(&arena.labels, labels_count * sizeof(int)));
  CC_CUDA_CHECK(cudaMalloc(&arena.labels_next, labels_count * sizeof(int)));
  CC_CUDA_CHECK(cudaMalloc(&arena.g, g_count * sizeof(GainT)));
  CC_CUDA_CHECK(cudaMalloc(&arena.g_next, g_count * sizeof(GainT)));
  CC_CUDA_CHECK(cudaMalloc(&arena.sizes, sizes_count * sizeof(int)));
  CC_CUDA_CHECK(cudaMalloc(&arena.sizes_next, sizes_count * sizeof(int)));
  CC_CUDA_CHECK(cudaMalloc(&arena.f, population * sizeof(long long)));
  CC_CUDA_CHECK(cudaMalloc(&arena.f_next, population * sizeof(long long)));
  if (masks_count > 0) CC_CUDA_CHECK(cudaMalloc(&arena.masks, masks_count * sizeof(uint32_t)));
  CC_CUDA_CHECK(cudaMalloc(&arena.moves, sizeof(unsigned long long)));
  CC_CUDA_CHECK(cudaMemset(arena.moves, 0, sizeof(unsigned long long)));
  CC_CUDA_CHECK(cudaMalloc(&arena.next_individual, sizeof(int)));
  if (use_k3) CC_CUDA_CHECK(cudaMalloc(&arena.work, labels_count * sizeof(unsigned long long)));

  // На широком пути сетка поэлементных ядер ограничена: индекс блока * размер блока считается в 32 битах, а
  // цикл с шагом сетки и так пройдёт все метки (поток ГПСЧ метки от сетки не зависит).
  const int init_blocks = (int)std::min<size_t>((labels_count + kBlockSize - 1) / kBlockSize,
                                                kWide ? (size_t)1 << 16 : (size_t)INT_MAX);
  KernelInitLabels<Index><<<init_blocks, kBlockSize>>>((Index)labels_count, k, arena.labels,
                                                       StreamSeed(params.seed, 0xA11CEull, 0));
  CC_CUDA_CHECK(cudaGetLastError());
  if (use_k2) {
    CC_CUDA_CHECK(cudaMalloc(&arena.degrees, n * sizeof(int)));
    KernelDegrees<<<(n + kBlockSize - 1) / kBlockSize, kBlockSize>>>(arena.bits, words, n, arena.degrees);
    CC_CUDA_CHECK(cudaGetLastError());
  }
  auto rebuild = [&]() {
    if (use_k2) {
      LaunchRebuild2(population, mask_bytes, arena.bits, words, n, arena.degrees, arena.labels, arena.g, arena.sizes,
                     arena.f, edges);
    } else {
      KernelRebuild<GainT><<<population, kBlockSize>>>(arena.bits, words, n, k, arena.labels, arena.g, arena.sizes,
                                                       arena.f, arena.masks, edges);
    }
    CC_CUDA_CHECK(cudaGetLastError());
  };
  rebuild();

  std::vector<long long> host_f(population);
  std::vector<int> host_labels(n);
  PbilsResult result;

  auto pull_best = [&](bool* improved) {
    CC_CUDA_CHECK(cudaMemcpy(host_f.data(), arena.f, population * sizeof(long long), cudaMemcpyDeviceToHost));
    int argmin = 0;
    for (int i = 1; i < population; ++i) {
      if (host_f[i] < host_f[argmin]) argmin = i;
    }
    if (result.labels.empty() || host_f[argmin] < result.objective) {
      result.objective = host_f[argmin];
      CC_CUDA_CHECK(cudaMemcpy(host_labels.data(), arena.labels + (size_t)argmin * n, n * sizeof(int),
                               cudaMemcpyDeviceToHost));
      result.labels = host_labels;
      if (improved != nullptr) *improved = true;
    }
  };

  pull_best(nullptr);

  int stall = 0;
  for (int iteration = 0; iteration < params.iterations; ++iteration) {
    KernelSelect<GainT><<<population, kBlockSize>>>(n, k, population, params.tournament, arena.labels, arena.g,
                                                    arena.sizes, arena.f, arena.labels_next, arena.g_next,
                                                    arena.sizes_next, arena.f_next,
                                                    StreamSeed(params.seed, 0x5E1EC7ull, 0), iteration);
    CC_CUDA_CHECK(cudaGetLastError());
    std::swap(arena.labels, arena.labels_next);
    std::swap(arena.g, arena.g_next);
    std::swap(arena.sizes, arena.sizes_next);
    std::swap(arena.f, arena.f_next);

    if (use_k2) {
      DispatchLocalSearch2(ls_blocks, population, arena.bits, words, n, arena.degrees, arena.labels, arena.g,
                           arena.sizes, arena.f, arena.moves, arena.next_individual);
    } else if (use_k3) {
      LaunchLocalSearch3(ls_blocks, population, arena.bits, words, n, arena.labels, arena.g, arena.sizes, arena.f,
                         arena.work, arena.moves, arena.next_individual);
    } else if (use_shared) {
      if constexpr (!kWide) {
        LaunchLocalSearch<true>(ls_threads, population, shared_bytes, arena.bits, words, n, k, arena.labels,
                                arena.g, arena.sizes, arena.f, arena.moves);
      }
    } else {
      LaunchLocalSearch<false>(ls_threads, population, 0, arena.bits, words, n, k, arena.labels, arena.g,
                               arena.sizes, arena.f, arena.moves);
    }
    CC_CUDA_CHECK(cudaGetLastError());

    bool improved = false;
    pull_best(&improved);
    result.local_searches += population;
    result.iterations_done = iteration + 1;
    stall = improved ? 0 : stall + 1;

    if (params.verbose) {
      std::printf("  iter %3d  record %lld  stall %d  %.2fs\n", iteration + 1, result.objective, stall,
                  SecondsSince(started));
    }
    if (stall >= params.early_stop) break;
    if (params.time_limit_sec > 0.0 && SecondsSince(started) >= params.time_limit_sec) break;

    if (k >= 2 && params.perturbation > 0.0) {
      KernelPerturb<Index><<<init_blocks, kBlockSize>>>((Index)labels_count, k, (float)params.perturbation,
                                                        arena.labels, StreamSeed(params.seed, 0xBEEFull, 0),
                                                        iteration);
      CC_CUDA_CHECK(cudaGetLastError());
      rebuild();
    }
  }

  unsigned long long moves = 0;
  CC_CUDA_CHECK(cudaMemcpy(&moves, arena.moves, sizeof(moves), cudaMemcpyDeviceToHost));
  CC_CUDA_CHECK(cudaDeviceSynchronize());

  result.accepted_moves = (long long)moves;
  result.seconds = SecondsSince(started);
  result.clusters_used = CountClustersUsed(result.labels, k);
  return result;
}

}  // namespace

PbilsResult SolveGpu(const Graph& graph, const PbilsParams& params) {
  if (params.k < 1) throw std::runtime_error("k must be >= 1");
  if (params.k > kMaxClusters) {
    throw std::runtime_error("k exceeds kMaxClusters (" + std::to_string(kMaxClusters) +
                             "); raise it in cc_gpu.hpp and rebuild");
  }
  if (params.population < 1) throw std::runtime_error("population must be >= 1");
  if ((int)graph.Size() > kMaxVerticesWide) {
    throw std::runtime_error("n exceeds " + std::to_string(kMaxVerticesWide) + ", the range of the packed move");
  }
  // Широкий путь включается сам, когда 16-битных G не хватает; --ls-kernel wide включает его и на малом
  // графе, чтобы оба пути можно было сравнить на одном инстансе.
  const bool wide = (int)graph.Size() > kMaxVerticesForGain || params.ls_kernel == PbilsParams::kLocalSearchWide;
  return wide ? Solve<WideGain>(graph, params) : Solve<Gain>(graph, params);
}

}  // namespace cc
