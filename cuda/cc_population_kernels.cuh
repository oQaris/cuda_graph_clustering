// Ядра над популяцией целиком: начальные метки, пересчёт G, селекция, GWW и возмущение.
#pragma once

#include "cc_gpu_common.cuh"
#include "cc_gpu_tuning.cuh"
#include "cc_graph.hpp"
#include "cc_math.hpp"

namespace cc::gpu {

using tuning::kBlockSize;

// Разбиения по окрестностям последовательных вершин: блок b отвечает за first + b. Собственную вершину включаем
// в кластер 0 явно, потому что диагональ матрицы смежности нулевая.
__global__ void KernelNeighborhoodLabels(const uint32_t* bits, int words, int n, int first, int* all_labels) {
  const int v = first + blockIdx.x;
  const uint32_t* row = bits + (size_t)v * words;
  int* labels = all_labels + (size_t)blockIdx.x * n;
  for (int u = threadIdx.x; u < n; u += blockDim.x) {
    labels[u] = (u == v || ((row[u / kWordBits] >> (u % kWordBits)) & 1u)) ? 0 : 1;
  }
}

// Поэлементные ядра проходят все population * n меток. Index — int на узком пути и long long на широком, где меток
// может быть больше 2^31.
template <typename Index>
__global__ void KernelInitLabels(Index total, int k, int* labels, uint64_t stream) {
  const Index stride = blockDim.x * gridDim.x;
  for (Index i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride) labels[i] = InitialLabel(stream, i, k);
}

template <typename Index>
__global__ void KernelPerturb(Index total, int k, float probability, int* labels, uint64_t stream, int iteration) {
  const Index stride = blockDim.x * gridDim.x;
  for (Index i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride) {
    const int label = labels[i];
    const int perturbed = PerturbedLabel(stream, i, iteration, label, k, probability);
    if (perturbed != label) labels[i] = perturbed;
  }
}

// Пересчёт G, размеров кластеров и f по меткам, блок на особь. G = A * Z побитово: строка смежности пересекается
// с маской каждого кластера, совпадения считает __popc.
template <typename GainT>
__global__ void KernelRebuild(const uint32_t* __restrict__ bits, int words, int n, int k, long long edges,
                              const int* __restrict__ all_labels, GainT* all_g, int* all_sizes, long long* f,
                              uint32_t* masks) {
  const int sol = blockIdx.x;
  const int tid = threadIdx.x;
  const int* labels = all_labels + (size_t)sol * n;
  GainT* g = all_g + (size_t)sol * k * n;
  int* sizes = all_sizes + (size_t)sol * k;
  uint32_t* my_masks = masks + (size_t)sol * k * words;

  for (int i = tid; i < k * words; i += kBlockSize) my_masks[i] = 0u;
  for (int c = tid; c < k; c += kBlockSize) sizes[c] = 0;
  __syncthreads();
  for (int v = tid; v < n; v += kBlockSize) {
    const int c = labels[v];
    atomicOr(&my_masks[(size_t)c * words + v / kWordBits], 1u << (v % kWordBits));
    atomicAdd(&sizes[c], 1);
  }
  __syncthreads();

  // Индексы в size_t, хотя помещаются в int: с int ptxas берёт 40 регистров вместо 48, и ядро медленнее (замер).
  long long intra_twice = 0;
  for (int v = tid; v < n; v += kBlockSize) {
    const uint32_t* row = bits + (size_t)v * words;
    for (int c = 0; c < k; ++c) {
      const uint32_t* mask = my_masks + (size_t)c * words;
      int acc = 0;
      for (int w = 0; w < words; ++w) acc += __popc(row[w] & mask[w]);
      g[(size_t)c * n + v] = (GainT)acc;
    }
    intra_twice += g[(size_t)labels[v] * n + v];
  }
  const long long sum = BlockSum<kBlockSize>(intra_twice);
  if (tid == 0) f[sol] = Objective(edges, sizes, k, sum / 2);
}

// Степени вершин: при k = 2 они заменяют половину G.
__global__ void KernelDegrees(const uint32_t* __restrict__ bits, int words, int n, int* degrees) {
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= n) return;
  const uint32_t* row = bits + (size_t)v * words;
  int acc = 0;
  for (int w = 0; w < words; ++w) acc += __popc(row[w]);
  degrees[v] = acc;
}

// Пересчёт при k = 2. Маска кластера 1 собирается __ballot_sync в разделяемой памяти (words слов): варп читает 32
// метки подряд — ровно слово маски. popcount нужен только по ней, G[0] = степень - G[1]: вдвое меньше проходов по
// матрице и ни одного атомика.
template <typename GainT>
__global__ void KernelRebuild2(const uint32_t* __restrict__ bits, int words, int n, long long edges,
                               const int* __restrict__ degrees, const int* __restrict__ all_labels, GainT* all_g,
                               int* all_sizes, long long* f) {
  const int sol = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid % kWarpSize;
  const int* labels = all_labels + (size_t)sol * n;
  GainT* g = all_g + (size_t)sol * 2 * n;
  uint32_t* mask = DynamicShared<uint32_t>();

  int ones = 0;
  for (int w = tid / kWarpSize; w < words; w += kBlockSize / kWarpSize) {
    const int v = w * kWarpSize + lane;
    const uint32_t word = __ballot_sync(kFullWarp, v < n && labels[v] != 0);
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
    g[v] = (GainT)g0;
    g[n + v] = (GainT)g1;
    intra_twice += (mask[v / kWordBits] >> (v % kWordBits)) & 1u ? g1 : g0;
  }

  // Обе суммы одной свёрткой: размер кластера 1 в старших битах, удвоенные внутренние рёбра (< n^2) в младших.
  constexpr int kCountShift = 2 * BitWidth(kMaxVerticesWide - 1);
  const long long sum = BlockSum<kBlockSize>(((long long)ones << kCountShift) + intra_twice);
  if (tid == 0) {
    int* sizes = all_sizes + (size_t)sol * 2;
    sizes[1] = (int)(sum >> kCountShift);
    sizes[0] = n - sizes[1];
    f[sol] = Objective(edges, sizes, 2, (sum & ((1LL << kCountShift) - 1)) / 2);
  }
}

// Турнирная селекция, блок на слот следующей популяции: метки, G и размеры победителя копируются целиком, так что
// пересчёт после неё не нужен.
template <typename GainT>
__global__ void KernelSelect(int n, int k, int population, int tournament, const int* __restrict__ labels,
                             const GainT* __restrict__ g, const int* __restrict__ sizes,
                             const long long* __restrict__ f, int* out_labels, GainT* out_g, int* out_sizes,
                             long long* out_f, uint64_t stream, int iteration) {
  const int slot = blockIdx.x;
  const int tid = threadIdx.x;

  __shared__ int winner;
  if (tid == 0) {
    winner = Tournament(stream, slot, iteration, f, population, tournament);
    out_f[slot] = f[winner];
  }
  __syncthreads();

  // Индексы считаются в каждом цикле заново: общие указатели держали бы живыми все шесть адресов сразу, и ядро
  // занимало бы на SM меньше блоков.
  const int source = winner;
  for (int v = tid; v < n; v += kBlockSize) out_labels[(size_t)slot * n + v] = labels[(size_t)source * n + v];
  for (int i = tid; i < k * n; i += kBlockSize) out_g[(size_t)slot * k * n + i] = g[(size_t)source * k * n + i];
  for (int c = tid; c < k; c += kBlockSize) out_sizes[(size_t)slot * k + c] = sizes[(size_t)source * k + c];
}

// GWW, блок на копию: худший слот order[population - 1 - i] получает решение лучшего order[i] целиком, вместе с G и f,
// потому что возмущения после копии может и не быть. Образцы и копии не пересекаются (kMaxGwwShare), копия — на месте.
template <typename GainT>
__global__ void KernelCopyWinners(int n, int k, int population, const int* __restrict__ order, int* labels, GainT* g,
                                  int* sizes, long long* f) {
  const int from = order[blockIdx.x];
  const int to = order[population - 1 - blockIdx.x];
  const int tid = threadIdx.x;
  if (tid == 0) f[to] = f[from];
  for (int v = tid; v < n; v += kBlockSize) labels[(size_t)to * n + v] = labels[(size_t)from * n + v];
  for (int i = tid; i < k * n; i += kBlockSize) g[(size_t)to * k * n + i] = g[(size_t)from * k * n + i];
  for (int c = tid; c < k; c += kBlockSize) sizes[(size_t)to * k + c] = sizes[(size_t)from * k + c];
}

}  // namespace cc::gpu
