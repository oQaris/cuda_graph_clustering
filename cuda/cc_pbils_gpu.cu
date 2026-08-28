// PBILS on CUDA.
//
// Parallel decomposition: one block per population member. Every member walks
// its own local search chain independently, so no cross-block synchronisation is
// needed inside an iteration, and the population size becomes the source of
// parallelism.
//
// Inside a block the threads split the vertex set: they evaluate all n*k moves
// against the maintained G matrix (G = A*Z, G[v][c] = neighbours of v inside
// cluster c), reduce to the steepest one, apply it, and repair G. Every
// arithmetic step comes from cc_math.hpp, the same header the CPU reference
// uses, so the two backends cannot drift.
//
// Where the time goes. A local search performs O(n) moves, and each move scans
// k*n entries of G and repairs 2*n of them. Held in global memory that is
// ~120 KB of traffic per move per solution, which on a measured run at n=6000
// pinned the memory controller at 98% while the arithmetic units idled. So the
// solution state lives in the block's shared memory whenever it fits there:
// the search then streams nothing but the adjacency row of the moved vertex,
// under a kilobyte per move. The two kernels implement the same steepest
// descent with the same tie-breaking, so they walk identical trajectories and
// must return identical objectives - which is what the tests check.
//
// G is stored as 16-bit integers: entries are counts bounded by n, so the range
// is ample, and halving the footprint both halves the traffic of the fallback
// kernel and doubles the instance size that fits in shared memory.
#include <cstdio>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

#include "cc_gpu.hpp"
#include "cc_math.hpp"

// Declaring dynamically sized shared memory, spelled so the host emulation in
// tools/cuda_host_emulation can compile this file too.
#if defined(CC_HOST_EMULATION)
#define CC_DYNAMIC_SHARED(type, name) \
  type* name = reinterpret_cast<type*>(shim::DynamicShared())
#else
#define CC_DYNAMIC_SHARED(type, name) extern __shared__ type name[]
#endif

namespace cc {
namespace {

constexpr int kBlock = 256;  // power of two: the reductions rely on it

// Entries of G are counts of neighbours, so they fit in 16 bits for any graph
// this solver targets. SolveGpu refuses larger instances rather than overflow.
using Gain = short;
constexpr int kMaxVerticesForGain = 32767;

#define CC_CUDA_CHECK(call)                                                     \
  do {                                                                          \
    const cudaError_t status = (call);                                          \
    if (status != cudaSuccess) {                                                \
      throw std::runtime_error(std::string("cuda error at ") + __FILE__ + ":" +  \
                               std::to_string(__LINE__) + ": " +                \
                               cudaGetErrorString(status));                     \
    }                                                                           \
  } while (0)

double SecondsSince(const std::chrono::steady_clock::time_point& start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// --------------------------------------------------------------------------
// Kernels
// --------------------------------------------------------------------------

__global__ void KernelInitLabels(int total, int k, int* labels, uint64_t seed) {
  const int stride = blockDim.x * gridDim.x;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
    uint64_t rng = StreamSeed(seed, static_cast<uint64_t>(idx), 0);
    labels[idx] = static_cast<int>(RandBelow(rng, static_cast<unsigned>(k)));
  }
}

// Recomputes G, cluster sizes and f from labels. G = A*Z is evaluated
// bit-parallel: each adjacency row is intersected with each cluster mask and
// counted with __popc, which is the unweighted-graph form of the matrix product.
__global__ void KernelRebuild(const uint32_t* __restrict__ bits, int words, int n, int k,
                              const int* __restrict__ labels, Gain* g, int* sizes,
                              long long* f, uint32_t* masks, long long edges) {
  const int sol = blockIdx.x;
  const int tid = threadIdx.x;

  const int* my_labels = labels + static_cast<size_t>(sol) * n;
  Gain* my_g = g + static_cast<size_t>(sol) * k * n;
  int* my_sizes = sizes + sol * k;
  uint32_t* my_masks = masks + static_cast<size_t>(sol) * k * words;

  for (int i = tid; i < k * words; i += kBlock) my_masks[i] = 0u;
  for (int c = tid; c < k; c += kBlock) my_sizes[c] = 0;
  __syncthreads();

  for (int v = tid; v < n; v += kBlock) {
    const int c = my_labels[v];
    atomicOr(&my_masks[static_cast<size_t>(c) * words + (v >> 5)], 1u << (v & 31));
    atomicAdd(&my_sizes[c], 1);
  }
  __syncthreads();

  long long intra_twice = 0;
  for (int v = tid; v < n; v += kBlock) {
    const uint32_t* row = bits + static_cast<size_t>(v) * words;
    for (int c = 0; c < k; ++c) {
      const uint32_t* mask = my_masks + static_cast<size_t>(c) * words;
      int acc = 0;
      for (int w = 0; w < words; ++w) acc += __popc(row[w] & mask[w]);
      my_g[static_cast<size_t>(c) * n + v] = static_cast<Gain>(acc);
    }
    intra_twice += my_g[static_cast<size_t>(my_labels[v]) * n + v];
  }

  __shared__ long long reduce[kBlock];
  reduce[tid] = intra_twice;
  __syncthreads();
  for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
    if (tid < stride) reduce[tid] += reduce[tid + stride];
    __syncthreads();
  }

  if (tid == 0) f[sol] = Objective(edges, my_sizes, k, reduce[0] / 2);
}

// Steepest descent over single-vertex moves, reading G from global memory.
// Used when the solution state is too large for shared memory.
__global__ void KernelLocalSearchGlobal(const uint32_t* __restrict__ bits, int words, int n,
                                        int k, int* labels, Gain* g, int* sizes, long long* f,
                                        unsigned long long* accepted_moves) {
  const int sol = blockIdx.x;
  const int tid = threadIdx.x;

  int* my_labels = labels + static_cast<size_t>(sol) * n;
  Gain* my_g = g + static_cast<size_t>(sol) * k * n;
  int* my_sizes = sizes + sol * k;

  __shared__ int cluster_size[kMaxClusters];
  __shared__ int best_delta[kBlock];
  __shared__ int best_vertex[kBlock];
  __shared__ int best_target[kBlock];
  __shared__ int move_vertex;
  __shared__ int move_from;
  __shared__ int move_to;
  __shared__ int move_delta;
  __shared__ long long objective;

  for (int c = tid; c < k; c += kBlock) cluster_size[c] = my_sizes[c];
  if (tid == 0) objective = f[sol];
  __syncthreads();

  unsigned long long applied = 0;
  while (true) {
    int local_delta = 0;
    int local_vertex = -1;
    int local_target = -1;

    for (int v = tid; v < n; v += kBlock) {
      const int from = my_labels[v];
      const int size_from = cluster_size[from];
      const int g_from = my_g[static_cast<size_t>(from) * n + v];
      for (int c = 0; c < k; ++c) {
        if (c == from) continue;
        const int delta = MoveDelta(size_from, cluster_size[c], g_from,
                                    my_g[static_cast<size_t>(c) * n + v]);
        if (delta < local_delta) {
          local_delta = delta;
          local_vertex = v;
          local_target = c;
        }
      }
    }

    best_delta[tid] = local_delta;
    best_vertex[tid] = local_vertex;
    best_target[tid] = local_target;
    __syncthreads();

    // Reduce to the steepest move, ties broken by the smaller vertex index so a
    // run is reproducible and both kernels agree.
    for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        const int other = tid + stride;
        const bool better = best_delta[other] < best_delta[tid] ||
                            (best_delta[other] == best_delta[tid] && best_vertex[other] >= 0 &&
                             (best_vertex[tid] < 0 || best_vertex[other] < best_vertex[tid]));
        if (better) {
          best_delta[tid] = best_delta[other];
          best_vertex[tid] = best_vertex[other];
          best_target[tid] = best_target[other];
        }
      }
      __syncthreads();
    }

    if (best_delta[0] >= 0 || best_vertex[0] < 0) break;

    if (tid == 0) {
      move_vertex = best_vertex[0];
      move_to = best_target[0];
      move_delta = best_delta[0];
      move_from = my_labels[move_vertex];
    }
    __syncthreads();

    const int v = move_vertex;
    const int from = move_from;
    const int to = move_to;
    const uint32_t* row = bits + static_cast<size_t>(v) * words;
    Gain* g_from = my_g + static_cast<size_t>(from) * n;
    Gain* g_to = my_g + static_cast<size_t>(to) * n;
    for (int u = tid; u < n; u += kBlock) {
      if ((row[u >> 5] >> (u & 31)) & 1u) {
        g_from[u] -= 1;
        g_to[u] += 1;
      }
    }

    if (tid == 0) {
      my_labels[v] = to;
      cluster_size[from] -= 1;
      cluster_size[to] += 1;
      objective += move_delta;
    }
    __syncthreads();
    ++applied;
  }

  for (int c = tid; c < k; c += kBlock) my_sizes[c] = cluster_size[c];
  if (tid == 0) {
    f[sol] = objective;
    if (accepted_moves != nullptr) atomicAdd(accepted_moves, applied);
  }
}

// The same steepest descent, but with G and the labels resident in the block's
// shared memory for the whole search. Requires 2*k*n + n bytes of dynamic
// shared memory; the host checks that before choosing this kernel.
__global__ void KernelLocalSearchShared(const uint32_t* __restrict__ bits, int words, int n,
                                        int k, int* labels, Gain* g, int* sizes, long long* f,
                                        unsigned long long* accepted_moves) {
  const int sol = blockIdx.x;
  const int tid = threadIdx.x;

  int* my_labels = labels + static_cast<size_t>(sol) * n;
  Gain* my_g = g + static_cast<size_t>(sol) * k * n;
  int* my_sizes = sizes + sol * k;

  CC_DYNAMIC_SHARED(unsigned char, dynamic_shared);
  Gain* sg = reinterpret_cast<Gain*>(dynamic_shared);            // k * n entries
  signed char* slabel = reinterpret_cast<signed char*>(sg + static_cast<size_t>(k) * n);

  __shared__ int cluster_size[kMaxClusters];
  __shared__ int best_delta[kBlock];
  __shared__ int best_vertex[kBlock];
  __shared__ int best_target[kBlock];
  __shared__ int move_vertex;
  __shared__ int move_from;
  __shared__ int move_to;
  __shared__ int move_delta;
  __shared__ long long objective;

  for (int i = tid; i < k * n; i += kBlock) sg[i] = my_g[i];
  for (int v = tid; v < n; v += kBlock) slabel[v] = static_cast<signed char>(my_labels[v]);
  for (int c = tid; c < k; c += kBlock) cluster_size[c] = my_sizes[c];
  if (tid == 0) objective = f[sol];
  __syncthreads();

  unsigned long long applied = 0;
  while (true) {
    int local_delta = 0;
    int local_vertex = -1;
    int local_target = -1;

    for (int v = tid; v < n; v += kBlock) {
      const int from = slabel[v];
      const int size_from = cluster_size[from];
      const int g_from = sg[static_cast<size_t>(from) * n + v];
      for (int c = 0; c < k; ++c) {
        if (c == from) continue;
        const int delta = MoveDelta(size_from, cluster_size[c], g_from,
                                    sg[static_cast<size_t>(c) * n + v]);
        if (delta < local_delta) {
          local_delta = delta;
          local_vertex = v;
          local_target = c;
        }
      }
    }

    best_delta[tid] = local_delta;
    best_vertex[tid] = local_vertex;
    best_target[tid] = local_target;
    __syncthreads();

    for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        const int other = tid + stride;
        const bool better = best_delta[other] < best_delta[tid] ||
                            (best_delta[other] == best_delta[tid] && best_vertex[other] >= 0 &&
                             (best_vertex[tid] < 0 || best_vertex[other] < best_vertex[tid]));
        if (better) {
          best_delta[tid] = best_delta[other];
          best_vertex[tid] = best_vertex[other];
          best_target[tid] = best_target[other];
        }
      }
      __syncthreads();
    }

    if (best_delta[0] >= 0 || best_vertex[0] < 0) break;

    if (tid == 0) {
      move_vertex = best_vertex[0];
      move_to = best_target[0];
      move_delta = best_delta[0];
      move_from = slabel[move_vertex];
    }
    __syncthreads();

    const int v = move_vertex;
    const int from = move_from;
    const int to = move_to;
    const uint32_t* row = bits + static_cast<size_t>(v) * words;
    Gain* g_from = sg + static_cast<size_t>(from) * n;
    Gain* g_to = sg + static_cast<size_t>(to) * n;
    for (int u = tid; u < n; u += kBlock) {
      if ((row[u >> 5] >> (u & 31)) & 1u) {
        g_from[u] -= 1;
        g_to[u] += 1;
      }
    }

    if (tid == 0) {
      slabel[v] = static_cast<signed char>(to);
      cluster_size[from] -= 1;
      cluster_size[to] += 1;
      objective += move_delta;
    }
    __syncthreads();
    ++applied;
  }

  for (int i = tid; i < k * n; i += kBlock) my_g[i] = sg[i];
  for (int v = tid; v < n; v += kBlock) my_labels[v] = slabel[v];
  for (int c = tid; c < k; c += kBlock) my_sizes[c] = cluster_size[c];
  if (tid == 0) {
    f[sol] = objective;
    if (accepted_moves != nullptr) atomicAdd(accepted_moves, applied);
  }
}

// Tournament selection. One block per slot of the next population; the winner's
// labels, G and sizes are copied over so no rebuild is needed afterwards.
__global__ void KernelSelect(int n, int k, int population, int tournament,
                             const int* __restrict__ labels, const Gain* __restrict__ g,
                             const int* __restrict__ sizes, const long long* __restrict__ f,
                             int* out_labels, Gain* out_g, int* out_sizes, long long* out_f,
                             uint64_t seed, int iteration) {
  const int slot = blockIdx.x;
  const int tid = threadIdx.x;

  __shared__ int winner;
  if (tid == 0) {
    uint64_t rng = StreamSeed(seed, static_cast<uint64_t>(slot), static_cast<uint64_t>(iteration) + 1);
    int chosen = static_cast<int>(RandBelow(rng, static_cast<unsigned>(population)));
    for (int t = 1; t < tournament; ++t) {
      const int candidate = static_cast<int>(RandBelow(rng, static_cast<unsigned>(population)));
      if (f[candidate] < f[chosen]) chosen = candidate;
    }
    winner = chosen;
    out_f[slot] = f[chosen];
  }
  __syncthreads();

  const int source = winner;
  for (int v = tid; v < n; v += kBlock) {
    out_labels[static_cast<size_t>(slot) * n + v] = labels[static_cast<size_t>(source) * n + v];
  }
  for (int i = tid; i < k * n; i += kBlock) {
    out_g[static_cast<size_t>(slot) * k * n + i] = g[static_cast<size_t>(source) * k * n + i];
  }
  for (int c = tid; c < k; c += kBlock) {
    out_sizes[slot * k + c] = sizes[source * k + c];
  }
}

// Relabels each vertex with the given probability; G is rebuilt afterwards.
__global__ void KernelPerturb(int total, int k, float probability, int* labels,
                              uint64_t seed, int iteration) {
  const int stride = blockDim.x * gridDim.x;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
    uint64_t rng = StreamSeed(seed, static_cast<uint64_t>(idx),
                              static_cast<uint64_t>(iteration) + 0x5000000ull);
    if (RandFloat(rng) >= probability) continue;
    const int shift = 1 + static_cast<int>(RandBelow(rng, static_cast<unsigned>(k - 1)));
    labels[idx] = (labels[idx] + shift) % k;
  }
}

// Device-side buffers for one run.
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
  if (static_cast<int>(graph.Size()) > kMaxVerticesForGain) {
    throw std::runtime_error("n exceeds " + std::to_string(kMaxVerticesForGain) +
                             ", the range of the 16-bit G matrix");
  }

  const auto started = std::chrono::steady_clock::now();

  const int n = static_cast<int>(graph.Size());
  const int k = params.k;
  const int population = params.population;
  const int words = static_cast<int>(graph.WordsPerRow());
  const long long edges = static_cast<long long>(graph.EdgeCount());

  // Choose where the solution state lives during the local search.
  const size_t shared_bytes = sizeof(Gain) * static_cast<size_t>(k) * n + static_cast<size_t>(n);
  int device = 0;
  CC_CUDA_CHECK(cudaGetDevice(&device));
  int shared_limit = 0;
  CC_CUDA_CHECK(cudaDeviceGetAttribute(&shared_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

  bool use_shared = false;
  if (params.ls_kernel == PbilsParams::kLocalSearchShared) {
    if (shared_bytes > static_cast<size_t>(shared_limit)) {
      throw std::runtime_error("state needs " + std::to_string(shared_bytes) +
                               " bytes of shared memory, device allows " +
                               std::to_string(shared_limit));
    }
    use_shared = true;
  } else if (params.ls_kernel == PbilsParams::kLocalSearchAuto) {
    use_shared = shared_bytes <= static_cast<size_t>(shared_limit);
  }

  if (use_shared) {
    CC_CUDA_CHECK(cudaFuncSetAttribute(KernelLocalSearchShared,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       static_cast<int>(shared_bytes)));
  }
  if (params.verbose) {
    std::printf("  local search state: %s (%zu bytes per block, device allows %d)\n",
                use_shared ? "shared memory" : "global memory", shared_bytes, shared_limit);
  }

  DeviceArena arena;
  const size_t bits_bytes = graph.Bits().size() * sizeof(uint32_t);
  const size_t labels_count = static_cast<size_t>(population) * n;
  const size_t g_count = static_cast<size_t>(population) * k * n;
  const size_t sizes_count = static_cast<size_t>(population) * k;
  const size_t masks_count = static_cast<size_t>(population) * k * words;

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

  const int init_blocks = static_cast<int>((labels_count + kBlock - 1) / kBlock);
  KernelInitLabels<<<init_blocks, kBlock>>>(static_cast<int>(labels_count), k, arena.labels,
                                            StreamSeed(params.seed, 0xA11CEull, 0));
  CC_CUDA_CHECK(cudaGetLastError());
  KernelRebuild<<<population, kBlock>>>(arena.bits, words, n, k, arena.labels, arena.g,
                                        arena.sizes, arena.f, arena.masks, edges);
  CC_CUDA_CHECK(cudaGetLastError());

  std::vector<long long> host_f(population);
  std::vector<int> host_labels(n);
  PbilsResult result;

  auto pull_best = [&](bool* improved) {
    CC_CUDA_CHECK(cudaMemcpy(host_f.data(), arena.f, population * sizeof(long long),
                             cudaMemcpyDeviceToHost));
    int argmin = 0;
    for (int i = 1; i < population; ++i) {
      if (host_f[i] < host_f[argmin]) argmin = i;
    }
    if (result.labels.empty() || host_f[argmin] < result.objective) {
      result.objective = host_f[argmin];
      CC_CUDA_CHECK(cudaMemcpy(host_labels.data(), arena.labels + static_cast<size_t>(argmin) * n,
                               n * sizeof(int), cudaMemcpyDeviceToHost));
      result.labels = host_labels;
      if (improved != nullptr) *improved = true;
    }
  };

  pull_best(nullptr);

  int stall = 0;
  for (int iteration = 0; iteration < params.iterations; ++iteration) {
    KernelSelect<<<population, kBlock>>>(n, k, population, params.tournament, arena.labels,
                                         arena.g, arena.sizes, arena.f, arena.labels_next,
                                         arena.g_next, arena.sizes_next, arena.f_next,
                                         StreamSeed(params.seed, 0x5E1EC7ull, 0), iteration);
    CC_CUDA_CHECK(cudaGetLastError());
    std::swap(arena.labels, arena.labels_next);
    std::swap(arena.g, arena.g_next);
    std::swap(arena.sizes, arena.sizes_next);
    std::swap(arena.f, arena.f_next);

    if (use_shared) {
      KernelLocalSearchShared<<<population, kBlock, shared_bytes>>>(
          arena.bits, words, n, k, arena.labels, arena.g, arena.sizes, arena.f, arena.moves);
    } else {
      KernelLocalSearchGlobal<<<population, kBlock>>>(
          arena.bits, words, n, k, arena.labels, arena.g, arena.sizes, arena.f, arena.moves);
    }
    CC_CUDA_CHECK(cudaGetLastError());

    bool improved = false;
    pull_best(&improved);
    result.local_searches += population;
    result.iterations_done = iteration + 1;
    stall = improved ? 0 : stall + 1;

    if (params.verbose) {
      std::printf("  iter %3d  record %lld  stall %d  %.2fs\n", iteration + 1, result.objective,
                  stall, SecondsSince(started));
    }
    if (stall >= params.early_stop) break;
    if (params.time_limit_sec > 0.0 && SecondsSince(started) >= params.time_limit_sec) break;

    if (k >= 2 && params.perturbation > 0.0) {
      KernelPerturb<<<init_blocks, kBlock>>>(static_cast<int>(labels_count), k,
                                             static_cast<float>(params.perturbation), arena.labels,
                                             StreamSeed(params.seed, 0xBEEFull, 0), iteration);
      CC_CUDA_CHECK(cudaGetLastError());
      KernelRebuild<<<population, kBlock>>>(arena.bits, words, n, k, arena.labels, arena.g,
                                            arena.sizes, arena.f, arena.masks, edges);
      CC_CUDA_CHECK(cudaGetLastError());
    }
  }

  unsigned long long moves = 0;
  CC_CUDA_CHECK(cudaMemcpy(&moves, arena.moves, sizeof(moves), cudaMemcpyDeviceToHost));
  CC_CUDA_CHECK(cudaDeviceSynchronize());

  result.accepted_moves = static_cast<long long>(moves);
  result.seconds = SecondsSince(started);
  result.clusters_used = CountClustersUsed(result.labels, k);
  return result;
}

}  // namespace cc
