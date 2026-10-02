// PBILS на CUDA: блок на особь, так что блоки между собой не синхронизируются, а параллелизм даёт сама популяция.
// Здесь выбор ядер, память и цикл итераций; ядра — в cc_*.cuh, параметры запуска — в cc_gpu_tuning.cuh.
//
// До 32 767 вершин G 16-битная (узкий путь). На большем графе решатель идёт широким путём: G 32-битная, а при k = 2
// и k = 3 спуск ведут ядра с упакованным состоянием вершины и постоянной сеткой под L2.
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "cc_gpu.hpp"
#include "cc_local_search.cuh"
#include "cc_local_search_k2.cuh"
#include "cc_local_search_k3.cuh"
#include "cc_population_kernels.cuh"

namespace cc {
namespace gpu {
namespace {

using tuning::kWideThreads;

struct Device {
  int sms = 0;
  int l2_bytes = 0;
  int shared_limit = 0;  // наибольшая разделяемая память на блок

  static Device Current() {
    int id = 0;
    CC_CUDA_CHECK(cudaGetDevice(&id));
    Device device;
    CC_CUDA_CHECK(cudaDeviceGetAttribute(&device.sms, cudaDevAttrMultiProcessorCount, id));
    CC_CUDA_CHECK(cudaDeviceGetAttribute(&device.l2_bytes, cudaDevAttrL2CacheSize, id));
    CC_CUDA_CHECK(cudaDeviceGetAttribute(&device.shared_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, id));
    return device;
  }
};

// Ядро спуска. Общее (kGlobal, kShared) годится для любого k; при k = 2 и k = 3 состояние вершины помещается в регистры
// или одно слово, и специальные ядра быстрее.
enum class Descent { kGlobal, kShared, kK2Registers, kK2Wide, kK3Wide };
constexpr const char* kDescentNames[] = {  // в порядке Descent, для --verbose
    "generic, global memory", "generic, shared memory", "k = 2, registers", "k = 2, packed words",
    "k = 3, packed words"};

struct Plan {
  Descent descent = Descent::kGlobal;
  int threads = 0;              // потоков на блок
  int blocks = 0;               // по блоку на особь или постоянная сетка
  int vertices_per_thread = 0;  // только kK2Registers
  size_t shared_bytes = 0;      // динамическая разделяемая память, только kShared

  bool K2() const { return descent == Descent::kK2Registers || descent == Descent::kK2Wide; }
};

// Блоков постоянной сетки: не больше, чем помещается на карту, и столько, чтобы состояние работающих особей
// (state_bytes на каждую) занимало долю l2_share кэша L2.
int ResidentBlocks(const void* kernel, size_t state_bytes, int population, double l2_share, const Device& device) {
  int per_sm = 0;
  CC_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kWideThreads, 0));
  const long long fit = (long long)(l2_share * device.l2_bytes / (double)std::max<size_t>(state_bytes, 1));
  const long long blocks = std::min<long long>(std::max(1, device.sms * per_sm), std::max(1LL, fit));
  return (int)std::min<long long>(blocks, population);
}

template <typename GainT>
Plan PlanDescent(int n, int k, const PbilsParams& params, const Device& device) {
  constexpr bool kWide = std::is_same_v<GainT, WideGain>;
  const int choice = params.ls_kernel;
  Plan plan;
  plan.blocks = params.population;
  plan.threads = tuning::GenericThreadsFor<GainT>(n, k, params.population);

  const size_t shared_bytes = sizeof(GainT) * (size_t)k * n + (size_t)n;
  const bool fits_shared = !kWide && shared_bytes <= (size_t)device.shared_limit;
  if (choice == PbilsParams::kLocalSearchShared && !fits_shared) {
    throw std::runtime_error("state needs " + std::to_string(shared_bytes) + " bytes of shared memory, device allows " +
                             std::to_string(device.shared_limit));
  }
  // Явный выбор shared или global оставляет общее ядро, чтобы с ним можно было сравнить специальные.
  const bool special = kWide ? choice != PbilsParams::kLocalSearchGlobal : choice == PbilsParams::kLocalSearchAuto;
  if (special && k == 2 && !kWide) {
    plan.descent = Descent::kK2Registers;
    plan.threads = tuning::K2ThreadsFor(n, params.population);
    plan.vertices_per_thread = tuning::K2VerticesPerThreadFor(n, plan.threads);
  } else if (special && k == 2) {
    plan.descent = Descent::kK2Wide;
    plan.blocks = ResidentBlocks((const void*)KernelLocalSearch2Wide<kWideThreads>, n * sizeof(int), params.population,
                                 tuning::kL2ShareK2, device);
  } else if (special && k == 3 && kWide) {
    plan.descent = Descent::kK3Wide;
    plan.blocks = ResidentBlocks((const void*)KernelLocalSearch3Wide<kWideThreads>, n * sizeof(unsigned long long),
                                 params.population, tuning::kL2ShareK3, device);
  } else if (choice != PbilsParams::kLocalSearchGlobal && fits_shared) {
    plan.descent = Descent::kShared;
    plan.shared_bytes = shared_bytes;
  }
  return plan;
}

// Память запуска на устройстве, освобождается разом.
class DeviceMemory {
 public:
  DeviceMemory() = default;
  DeviceMemory(const DeviceMemory&) = delete;
  DeviceMemory& operator=(const DeviceMemory&) = delete;
  ~DeviceMemory() {
    for (void* block : blocks_) cudaFree(block);
  }

  template <typename T>
  void Allocate(T*& ptr, size_t count) {
    CC_CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    blocks_.push_back(ptr);
  }

 private:
  std::vector<void*> blocks_;
};

std::string Mebibytes(size_t bytes) { return std::to_string(CeilDiv(bytes, 1 << 20)) + " MiB"; }

// Разрешает ядру динамическую разделяемую память сверх умолчания (48 КБ).
template <typename Kernel>
void AllowDynamicShared(Kernel* kernel, size_t bytes) {
  CC_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes));
}

// Популяция на устройстве: метки, G, размеры кластеров и f всех особей подряд. Ядрам поля передаются по отдельности, а
// не этой структурой: с ней NVVM держит базовые адреса в регистрах весь спуск, и ядро k = 2 при n = 4000 медленнее в
// 1,5 раза (замер).
template <typename GainT>
struct Population {
  int n = 0;
  int k = 0;
  int* labels = nullptr;  // n на особь
  GainT* g = nullptr;     // k * n на особь, строка кластера c — с c * n
  int* sizes = nullptr;   // k на особь
  long long* f = nullptr;
};

// Всё, что нужно ядрам одного запуска.
template <typename GainT>
struct Run {
  int words = 0;
  int population = 0;
  int copies = 0;  // сколько худших особей GWW заменяет копиями лучших
  long long edges = 0;
  Plan plan;
  DeviceMemory memory;
  uint32_t* bits = nullptr;
  int* degrees = nullptr;               // степени вершин, для ядер k = 2
  uint32_t* masks = nullptr;            // маски кластеров KernelRebuild
  unsigned long long* work = nullptr;   // слова вершин KernelLocalSearch3Wide
  unsigned long long* moves = nullptr;  // принятые ходы, для --verbose
  int* next_individual = nullptr;       // счётчик постоянной сетки
  int* order = nullptr;                 // слоты от лучшего к худшему, для GWW
  Population<GainT> pop;                // текущая популяция
  Population<GainT> next;               // сюда пишет селекция

  // f(указатель, число элементов) для каждого буфера; 0 элементов — этому плану буфер не нужен. Один список даёт и
  // объём, который сверяется со свободной памятью, и выделение.
  template <typename F>
  void ForEachBuffer(size_t bits_words, F&& f) {
    const size_t labels = (size_t)population * pop.n;
    f(bits, bits_words);
    f(degrees, plan.K2() ? (size_t)pop.n : 0);
    f(masks, plan.K2() ? 0 : (size_t)population * pop.k * words);
    f(work, plan.descent == Descent::kK3Wide ? labels : 0);
    f(moves, 1);
    f(next_individual, 1);
    f(order, copies > 0 ? (size_t)population : 0);
    for (Population<GainT>* p : {&pop, &next}) {
      f(p->labels, labels);
      f(p->g, labels * pop.k);
      f(p->sizes, (size_t)population * pop.k);
      f(p->f, (size_t)population);
    }
  }

  Run(const Graph& graph, const PbilsParams& params)
      : words((int)graph.WordsPerRow()),
        population(params.population),
        copies(WinnerCopies(params.gww, params.population)) {
    const Device device = Device::Current();
    const int n = (int)graph.Size();
    edges = (long long)graph.EdgeCount();
    plan = PlanDescent<GainT>(n, params.k, params, device);
    pop.n = next.n = n;
    pop.k = next.k = params.k;

    // Под WDDM выделение сверх памяти карты проходит: драйвер молча выносит её в ОЗУ, и ядра замедляются в разы.
    size_t bytes = 0;
    ForEachBuffer(graph.Bits().size(), [&](auto*& ptr, size_t count) { bytes += count * sizeof(*ptr); });
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CC_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (bytes > free_bytes) {
      throw std::runtime_error("run needs " + Mebibytes(bytes) + " of device memory, " + Mebibytes(free_bytes) +
                               " free");
    }
    ForEachBuffer(graph.Bits().size(), [&](auto*& ptr, size_t count) {
      if (count > 0) memory.Allocate(ptr, count);
    });
    const std::vector<Word>& host_bits = graph.Bits();
    CC_CUDA_CHECK(cudaMemcpy(bits, host_bits.data(), host_bits.size() * sizeof(Word), cudaMemcpyHostToDevice));
    CC_CUDA_CHECK(cudaMemset(moves, 0, sizeof(*moves)));
    ConfigureSharedMemory(device);
    if (plan.K2()) {
      KernelDegrees<<<(int)CeilDiv(n, tuning::kBlockSize), tuning::kBlockSize>>>(bits, words, n, degrees);
      CC_CUDA_CHECK(cudaGetLastError());
    }
    if (params.verbose) {
      std::printf("  local search: %s, %d threads per block, %d blocks", kDescentNames[(int)plan.descent], plan.threads,
                  plan.blocks);
      if (plan.descent == Descent::kK2Registers) std::printf(", %d vertices per thread", plan.vertices_per_thread);
      if (plan.descent == Descent::kShared) std::printf(", %zu bytes of shared memory", plan.shared_bytes);
      std::printf("\n  device memory: %s of %s free\n", Mebibytes(bytes).c_str(), Mebibytes(free_bytes).c_str());
    }
  }

  size_t MaskBytes() const { return words * sizeof(uint32_t); }  // маска кластера 1 у KernelRebuild2

  // Динамическая разделяемая память сверх умолчания: маска у пересчёта k = 2, состояние у kShared.
  void ConfigureSharedMemory(const Device& device) {
    if (plan.K2()) {
      if (MaskBytes() > (size_t)device.shared_limit) {
        throw std::runtime_error("cluster mask needs " + std::to_string(MaskBytes()) +
                                 " bytes of shared memory, device allows " + std::to_string(device.shared_limit));
      }
      AllowDynamicShared(KernelRebuild2<GainT>, MaskBytes());
    }
    if constexpr (std::is_same_v<GainT, Gain>) {
      if (plan.descent != Descent::kShared) return;
      WithConstant(plan.threads, tuning::GenericThreads<Gain>{}, [&](auto threads) {
        AllowDynamicShared(KernelLocalSearch<true, decltype(threads)::value, Gain>, plan.shared_bytes);
      });
    }
  }
};

void CheckLaunch() { CC_CUDA_CHECK(cudaGetLastError()); }

template <typename GainT>
void Rebuild(const Run<GainT>& run) {
  const Population<GainT>& p = run.pop;
  if (run.plan.K2()) {
    KernelRebuild2<GainT><<<run.population, tuning::kBlockSize, run.MaskBytes()>>>(
        run.bits, run.words, p.n, run.edges, run.degrees, p.labels, p.g, p.sizes, p.f);
  } else {
    KernelRebuild<GainT><<<run.population, tuning::kBlockSize>>>(run.bits, run.words, p.n, p.k, run.edges, p.labels,
                                                                 p.g, p.sizes, p.f, run.masks);
  }
  CheckLaunch();
}

template <typename GainT>
void Select(Run<GainT>& run, int tournament, uint64_t stream, int iteration) {
  const Population<GainT>& p = run.pop;
  const Population<GainT>& q = run.next;
  KernelSelect<GainT><<<run.population, tuning::kBlockSize>>>(p.n, p.k, run.population, tournament, p.labels, p.g,
                                                              p.sizes, p.f, q.labels, q.g, q.sizes, q.f, stream,
                                                              iteration);
  CheckLaunch();
  std::swap(run.pop, run.next);
}

template <bool kShared, typename GainT>
void DescendGeneric(const Run<GainT>& run) {
  const Population<GainT>& p = run.pop;
  WithConstant(run.plan.threads, tuning::GenericThreads<GainT>{}, [&](auto threads) {
    constexpr int kThreads = decltype(threads)::value;
    KernelLocalSearch<kShared, kThreads, GainT><<<run.population, kThreads, run.plan.shared_bytes>>>(
        run.bits, run.words, p.n, p.k, p.labels, p.g, p.sizes, p.f, run.moves);
  });
}

// Специальные ядра есть только на одном из путей, поэтому их запуск разведён перегрузками по типу G: nvcc не всегда
// отбрасывает ветку if constexpr с запуском ядра внутри лямбды.
void DescendSpecial(const Run<Gain>& run) {
  const Population<Gain>& p = run.pop;
  WithConstant(run.plan.threads, tuning::K2Threads{}, [&](auto threads) {
    WithConstant(run.plan.vertices_per_thread, tuning::K2VerticesPerThread{}, [&](auto vertices) {
      constexpr int kThreads = decltype(threads)::value;
      KernelLocalSearch2<kThreads, decltype(vertices)::value>
          <<<run.population, kThreads>>>(run.bits, run.words, p.n, p.labels, p.g, p.sizes, p.f, run.moves);
    });
  });
}

void DescendSpecial(const Run<WideGain>& run) {
  const Population<WideGain>& p = run.pop;
  CC_CUDA_CHECK(cudaMemsetAsync(run.next_individual, 0, sizeof(int)));
  if (run.plan.descent == Descent::kK2Wide) {
    KernelLocalSearch2Wide<kWideThreads><<<run.plan.blocks, kWideThreads>>>(run.bits, run.words, p.n, run.population,
                                                                            run.degrees, p.labels, p.g, p.sizes, p.f,
                                                                            run.moves, run.next_individual);
  } else {
    KernelLocalSearch3Wide<kWideThreads><<<run.plan.blocks, kWideThreads>>>(run.bits, run.words, p.n, run.population,
                                                                            p.labels, p.g, p.sizes, p.f, run.work,
                                                                            run.moves, run.next_individual);
  }
}

template <typename GainT>
void Descend(const Run<GainT>& run) {
  if (run.plan.descent == Descent::kGlobal) {
    DescendGeneric<false>(run);
  } else if (run.plan.descent == Descent::kShared) {
    if constexpr (std::is_same_v<GainT, Gain>) DescendGeneric<true>(run);
  } else {
    DescendSpecial(run);
  }
  CheckLaunch();
}

// Лучшая особь популяции; если она лучше рекорда, рекорд берёт её метки. true — рекорд улучшен.
template <typename GainT>
bool PullRecord(const Run<GainT>& run, std::vector<long long>& host_f, PbilsResult& result) {
  CC_CUDA_CHECK(cudaMemcpy(host_f.data(), run.pop.f, host_f.size() * sizeof(long long), cudaMemcpyDeviceToHost));
  const int best = (int)(std::min_element(host_f.begin(), host_f.end()) - host_f.begin());
  if (!result.labels.empty() && host_f[best] >= result.objective) return false;
  result.objective = host_f[best];
  result.labels.resize(run.pop.n);
  const int* labels = run.pop.labels + (size_t)best * run.pop.n;
  CC_CUDA_CHECK(cudaMemcpy(result.labels.data(), labels, run.pop.n * sizeof(int), cudaMemcpyDeviceToHost));
  return true;
}

// GWW: порядок слотов считает хост по f, которые PullRecord уже скопировал, — тот же, что у CPU.
template <typename GainT>
void CopyWinners(const Run<GainT>& run, const std::vector<long long>& host_f) {
  const std::vector<int> order = RankSlots(host_f.data(), run.population);
  CC_CUDA_CHECK(cudaMemcpy(run.order, order.data(), order.size() * sizeof(int), cudaMemcpyHostToDevice));
  const Population<GainT>& p = run.pop;
  KernelCopyWinners<GainT>
      <<<run.copies, tuning::kBlockSize>>>(p.n, p.k, run.population, run.order, p.labels, p.g, p.sizes, p.f);
  CheckLaunch();
}

template <typename GainT>
PbilsResult Solve(const Graph& graph, const PbilsParams& params) {
  using Index = std::conditional_t<std::is_same_v<GainT, WideGain>, long long, int>;
  Progress progress(params);
  Run<GainT> run(graph, params);
  const int k = params.k;
  const Index labels = (Index)params.population * run.pop.n;
  const int grid = (int)std::min(CeilDiv(labels, tuning::kBlockSize), tuning::kMaxGridBlocks);
  const uint64_t init_stream = StreamSeed(params.seed, kInitStream, 0);
  const uint64_t select_stream = StreamSeed(params.seed, kSelectStream, 0);
  const uint64_t perturb_stream = StreamSeed(params.seed, kPerturbStream, 0);

  KernelInitLabels<Index><<<grid, tuning::kBlockSize>>>(labels, k, run.pop.labels, init_stream);
  CheckLaunch();
  Rebuild(run);

  PbilsResult result;
  std::vector<long long> host_f(params.population);
  PullRecord(run, host_f, result);
  for (int iteration = 0; iteration < params.iterations; ++iteration) {
    Select(run, params.tournament, select_stream, iteration);
    Descend(run);
    if (progress.Next(result, PullRecord(run, host_f, result))) break;
    if (run.copies > 0) CopyWinners(run, host_f);

    if (k >= 2 && params.perturbation > 0.0) {
      KernelPerturb<Index><<<grid, tuning::kBlockSize>>>(labels, k, (float)params.perturbation, run.pop.labels,
                                                         perturb_stream, iteration);
      CheckLaunch();
      Rebuild(run);
    }
  }

  unsigned long long moves = 0;
  CC_CUDA_CHECK(cudaMemcpy(&moves, run.moves, sizeof(moves), cudaMemcpyDeviceToHost));
  CC_CUDA_CHECK(cudaDeviceSynchronize());
  result.accepted_moves = (long long)moves;
  result.seconds = progress.Seconds();
  result.clusters_used = CountClustersUsed(result.labels, k);
  return result;
}

}  // namespace
}  // namespace gpu

PbilsResult SolveGpu(const Graph& graph, const PbilsParams& params) {
  CheckParams(params);
  if (params.k > kMaxClusters) {
    throw std::runtime_error("k exceeds kMaxClusters (" + std::to_string(kMaxClusters) +
                             "); raise it in cc_gpu.hpp and rebuild");
  }
  const int n = (int)graph.Size();
  if (n > gpu::kMaxVerticesWide) {
    throw std::runtime_error("n exceeds " + std::to_string(gpu::kMaxVerticesWide) + ", the range of the packed move");
  }
  // --ls-kernel wide включает широкий путь и на малом графе, чтобы оба пути можно было сравнить на одном инстансе.
  const bool wide = n > gpu::kMaxVerticesNarrow || params.ls_kernel == PbilsParams::kLocalSearchWide;
  return wide ? gpu::Solve<gpu::WideGain>(graph, params) : gpu::Solve<gpu::Gain>(graph, params);
}

}  // namespace cc
