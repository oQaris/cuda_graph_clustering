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
#include <cstdio>
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

// Элементы G — счётчики соседей, не больше n, поэтому 16 бит хватает с запасом для любого
// инстанса, с которым имеет дело решатель; SolveGpu отказывает на больших графах, а не
// переполняется молча.
using Gain = short;
constexpr int kMaxVerticesForGain = 32767;

// Ход в KernelLocalSearch упакован в 31 бит: вершина << 12 | откуда << 6 | куда.
static_assert(kMaxClusters <= 64, "cluster index must fit in 6 bits of the packed move");
static_assert(kMaxVerticesForGain < (1 << 19), "vertex index must fit in 19 bits of the packed move");

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

__global__ void KernelInitLabels(int total, int k, int* labels, uint64_t seed) {
  const int stride = blockDim.x * gridDim.x;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
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
__global__ void KernelRebuild(const uint32_t* __restrict__ bits, int words, int n, int k,
                              const int* __restrict__ labels, Gain* g, int* sizes, long long* f,
                              uint32_t* masks, long long edges) {
  const int sol = blockIdx.x;
  const int tid = threadIdx.x;

  const int* my_labels = labels + (size_t)sol * n;
  Gain* my_g = g + (size_t)sol * k * n;
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
      my_g[(size_t)c * n + v] = (Gain)acc;
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
template <bool kShared, int kThreads>
__global__ void __launch_bounds__(kThreads)
KernelLocalSearch(const uint32_t* __restrict__ bits, int words, int n, int k, int* labels, Gain* g,
                  int* sizes, long long* f, unsigned long long* accepted_moves) {
  using Label = typename std::conditional<kShared, signed char, int>::type;
  constexpr int kWarps = kThreads / 32;
  static_assert(kThreads % 32 == 0, "block must be whole warps");

  const int sol = blockIdx.x;
  const int tid = threadIdx.x;

  int* my_labels = labels + (size_t)sol * n;
  Gain* my_g = g + (size_t)sol * k * n;
  int* my_sizes = sizes + (size_t)sol * k;

  __shared__ long long warp_best[2][kWarps];

  Gain* wg;   // рабочее G текущего решения
  Label* wl;  // рабочие метки текущего решения
  if constexpr (kShared) {
    extern __shared__ unsigned char dynamic_shared[];
    Gain* sg = reinterpret_cast<Gain*>(dynamic_shared);            // k * n элементов
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
            (long long)((local_vertex << 12) | (local_from << 6) | local_target);
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
    const int payload = (int)(best & 0xffffffffLL);
    const int v = payload >> 12;
    const int from = (payload >> 6) & 63;
    const int to = payload & 63;

    // Буфер parity дочитан всеми (они прошли барьер): доводим его до размеров после этого хода.
    for (int c = tid; c < k; c += kThreads) {
      cluster_size[parity][c] += (c == prev_to) - (c == prev_from) + (c == to) - (c == from);
    }
    prev_from = from;
    prev_to = to;
    parity ^= 1;

    const uint32_t* row = bits + (size_t)v * words;
    Gain* g_from = wg + from * n;
    Gain* g_to = wg + to * n;
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

// Потоков на блок локального поиска. Правило снято замером на RTX 4070 Ti SUPER (66 SM), сетка
// n = 500..16000, k = 3 и 8, популяция 64..4096. Пока блоков мало, карта недогружена, и больший блок
// быстрее — до 1,95x при популяции 64. Когда популяция уже заполняет SM, а работы на особь мало
// (k*n < 10000), выгоднее 256: у 512 потоков остаётся по несколько вершин на поток, и ход съедает
// синхронизация — там 256 быстрее до 1,6x. В остальных точках разница в пределах нескольких процентов.
int LocalSearchThreads(int n, int k, int population) {
  return (population >= 256 && (long long)k * n < 10000) ? 256 : 512;
}

template <bool kShared>
void LaunchLocalSearch(int threads, int population, size_t shared_bytes, const uint32_t* bits, int words,
                       int n, int k, int* labels, Gain* g, int* sizes, long long* f,
                       unsigned long long* moves) {
  if (threads == 256) {
    KernelLocalSearch<kShared, 256><<<population, 256, shared_bytes>>>(bits, words, n, k, labels, g, sizes, f,
                                                                       moves);
  } else {
    KernelLocalSearch<kShared, 512><<<population, 512, shared_bytes>>>(bits, words, n, k, labels, g, sizes, f,
                                                                       moves);
  }
}

// Турнирная селекция. Один блок на слот следующей популяции; метки, G и размеры победителя
// копируются целиком, поэтому пересчёт после неё не нужен.
__global__ void KernelSelect(int n, int k, int population, int tournament,
                             const int* __restrict__ labels, const Gain* __restrict__ g,
                             const int* __restrict__ sizes, const long long* __restrict__ f,
                             int* out_labels, Gain* out_g, int* out_sizes, long long* out_f,
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
__global__ void KernelPerturb(int total, int k, float probability, int* labels, uint64_t seed,
                              int iteration) {
  const int stride = blockDim.x * gridDim.x;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
    uint64_t rng = StreamSeed(seed, idx, iteration + 0x5000000ull);
    if (RandFloat(rng) >= probability) continue;
    const int shift = 1 + (int)RandBelow(rng, k - 1);
    labels[idx] = (labels[idx] + shift) % k;
  }
}

// Буферы устройства для одного запуска.
struct DeviceArena {
  uint32_t* bits = nullptr;
  int* labels = nullptr;
  int* labels_next = nullptr;
  Gain* g = nullptr;
  Gain* g_next = nullptr;
  int* sizes = nullptr;
  int* sizes_next = nullptr;
  long long* f = nullptr;
  long long* f_next = nullptr;
  uint32_t* masks = nullptr;
  unsigned long long* moves = nullptr;

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
  }
};

}  // namespace

PbilsResult SolveGpu(const Graph& graph, const PbilsParams& params) {
  if (params.k < 1) throw std::runtime_error("k must be >= 1");
  if (params.k > kMaxClusters) {
    throw std::runtime_error("k exceeds kMaxClusters (" + std::to_string(kMaxClusters) +
                             "); raise it in cc_gpu.hpp and rebuild");
  }
  if (params.population < 1) throw std::runtime_error("population must be >= 1");
  if ((int)graph.Size() > kMaxVerticesForGain) {
    throw std::runtime_error("n exceeds " + std::to_string(kMaxVerticesForGain) +
                             ", the range of the 16-bit G matrix");
  }

  const auto started = std::chrono::steady_clock::now();

  const int n = (int)graph.Size();
  const int k = params.k;
  const int population = params.population;
  const int words = (int)graph.WordsPerRow();
  const long long edges = (long long)graph.EdgeCount();

  // Выбор, где состояние решения живёт во время локального поиска.
  const size_t shared_bytes = sizeof(Gain) * (size_t)k * n + (size_t)n;
  int device = 0;
  CC_CUDA_CHECK(cudaGetDevice(&device));
  int shared_limit = 0;
  CC_CUDA_CHECK(cudaDeviceGetAttribute(&shared_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

  bool use_shared = false;
  if (params.ls_kernel == PbilsParams::kLocalSearchShared) {
    if (shared_bytes > (size_t)shared_limit) {
      throw std::runtime_error("state needs " + std::to_string(shared_bytes) +
                               " bytes of shared memory, device allows " + std::to_string(shared_limit));
    }
    use_shared = true;
  } else if (params.ls_kernel == PbilsParams::kLocalSearchAuto) {
    use_shared = shared_bytes <= (size_t)shared_limit;
  }

  if (use_shared) {
    CC_CUDA_CHECK(cudaFuncSetAttribute(KernelLocalSearch<true, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       (int)shared_bytes));
    CC_CUDA_CHECK(cudaFuncSetAttribute(KernelLocalSearch<true, 512>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       (int)shared_bytes));
  }
  const int ls_threads = LocalSearchThreads(n, k, population);
  if (params.verbose) {
    std::printf("  local search state: %s (%zu bytes per block, device allows %d), %d threads per block\n",
                use_shared ? "shared memory" : "global memory", shared_bytes, shared_limit, ls_threads);
  }

  DeviceArena arena;
  const size_t bits_bytes = graph.Bits().size() * sizeof(uint32_t);
  const size_t labels_count = (size_t)population * n;
  const size_t g_count = (size_t)population * k * n;
  const size_t sizes_count = (size_t)population * k;
  const size_t masks_count = (size_t)population * k * words;

  CC_CUDA_CHECK(cudaMalloc(&arena.bits, bits_bytes));
  CC_CUDA_CHECK(cudaMemcpy(arena.bits, graph.Bits().data(), bits_bytes, cudaMemcpyHostToDevice));
  CC_CUDA_CHECK(cudaMalloc(&arena.labels, labels_count * sizeof(int)));
  CC_CUDA_CHECK(cudaMalloc(&arena.labels_next, labels_count * sizeof(int)));
  CC_CUDA_CHECK(cudaMalloc(&arena.g, g_count * sizeof(Gain)));
  CC_CUDA_CHECK(cudaMalloc(&arena.g_next, g_count * sizeof(Gain)));
  CC_CUDA_CHECK(cudaMalloc(&arena.sizes, sizes_count * sizeof(int)));
  CC_CUDA_CHECK(cudaMalloc(&arena.sizes_next, sizes_count * sizeof(int)));
  CC_CUDA_CHECK(cudaMalloc(&arena.f, population * sizeof(long long)));
  CC_CUDA_CHECK(cudaMalloc(&arena.f_next, population * sizeof(long long)));
  CC_CUDA_CHECK(cudaMalloc(&arena.masks, masks_count * sizeof(uint32_t)));
  CC_CUDA_CHECK(cudaMalloc(&arena.moves, sizeof(unsigned long long)));
  CC_CUDA_CHECK(cudaMemset(arena.moves, 0, sizeof(unsigned long long)));

  const int init_blocks = (int)((labels_count + kBlockSize - 1) / kBlockSize);
  KernelInitLabels<<<init_blocks, kBlockSize>>>((int)labels_count, k, arena.labels,
                                                StreamSeed(params.seed, 0xA11CEull, 0));
  CC_CUDA_CHECK(cudaGetLastError());
  KernelRebuild<<<population, kBlockSize>>>(arena.bits, words, n, k, arena.labels, arena.g, arena.sizes,
                                            arena.f, arena.masks, edges);
  CC_CUDA_CHECK(cudaGetLastError());

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
    KernelSelect<<<population, kBlockSize>>>(n, k, population, params.tournament, arena.labels, arena.g,
                                             arena.sizes, arena.f, arena.labels_next, arena.g_next,
                                             arena.sizes_next, arena.f_next,
                                             StreamSeed(params.seed, 0x5E1EC7ull, 0), iteration);
    CC_CUDA_CHECK(cudaGetLastError());
    std::swap(arena.labels, arena.labels_next);
    std::swap(arena.g, arena.g_next);
    std::swap(arena.sizes, arena.sizes_next);
    std::swap(arena.f, arena.f_next);

    if (use_shared) {
      LaunchLocalSearch<true>(ls_threads, population, shared_bytes, arena.bits, words, n, k, arena.labels,
                              arena.g, arena.sizes, arena.f, arena.moves);
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
      KernelPerturb<<<init_blocks, kBlockSize>>>((int)labels_count, k, (float)params.perturbation,
                                                 arena.labels, StreamSeed(params.seed, 0xBEEFull, 0),
                                                 iteration);
      CC_CUDA_CHECK(cudaGetLastError());
      KernelRebuild<<<population, kBlockSize>>>(arena.bits, words, n, k, arena.labels, arena.g, arena.sizes,
                                                arena.f, arena.masks, edges);
      CC_CUDA_CHECK(cudaGetLastError());
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

}  // namespace cc
