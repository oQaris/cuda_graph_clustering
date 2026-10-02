// Общее для ядер: проверка ошибок CUDA, популяция на устройстве, свёртки по варпу и блоку, ключ хода.
#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "cc_gpu.hpp"

#define CC_CUDA_CHECK(call)                                                                                       \
  do {                                                                                                            \
    const cudaError_t status = (call);                                                                            \
    if (status != cudaSuccess) {                                                                                  \
      throw std::runtime_error(std::string("cuda error at ") + __FILE__ + ":" + std::to_string(__LINE__) + ": " + \
                               cudaGetErrorString(status));                                                       \
    }                                                                                                             \
  } while (0)

namespace cc::gpu {

constexpr int kWarpSize = 32;
constexpr unsigned kFullWarp = 0xffffffffu;

// Элементы G — счётчики соседей, не больше n - 1. Узкий путь держит их в 16 битах: вдвое меньше трафика, и вдвое
// больший граф помещается в разделяемую память. На большем графе G 32-битная — широкий путь.
using Gain = short;
using WideGain = int;
constexpr int kMaxVerticesNarrow = std::numeric_limits<Gain>::max();
constexpr int kMaxVerticesWide = 1 << 20;  // предел упакованного хода, см. PackedMove

// Сколько бит нужно для чисел из [0, x].
__host__ __device__ constexpr int BitWidth(unsigned x) { return x == 0 ? 0 : 1 + BitWidth(x >> 1); }

constexpr long long CeilDiv(long long a, long long b) { return (a + b - 1) / b; }

// Динамическая разделяемая память блока. Имя и тип у всех ядер одни: extern __shared__ разных типов не совместимы.
template <typename T>
__device__ __forceinline__ T* DynamicShared() {
  extern __shared__ unsigned char dynamic_shared[];
  return reinterpret_cast<T*>(dynamic_shared);
}

// Минимум по варпу, результат у всех потоков. На sm_80+ — аппаратная редукция, 64-битный ключ — в две: сначала
// старшая половина, затем младшая среди потоков с минимальной старшей.
__device__ __forceinline__ unsigned WarpMin(unsigned key) {
#if __CUDA_ARCH__ >= 800
  return __reduce_min_sync(kFullWarp, key);
#else
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) key = min(key, __shfl_xor_sync(kFullWarp, key, offset));
  return key;
#endif
}

__device__ __forceinline__ unsigned long long WarpMin(unsigned long long key) {
#if __CUDA_ARCH__ >= 800
  const unsigned high = __reduce_min_sync(kFullWarp, (unsigned)(key >> 32));
  const unsigned low = __reduce_min_sync(kFullWarp, (unsigned)(key >> 32) == high ? (unsigned)key : 0xffffffffu);
  return ((unsigned long long)high << 32) | low;
#else
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    const unsigned long long other = __shfl_xor_sync(kFullWarp, key, offset);
    if (other < key) key = other;
  }
  return key;
#endif
}

// Минимум по блоку за один барьер: варпы сворачивают ключи сами, итог по варпам досчитывает каждый поток. partial —
// половина двойного буфера, которую ходы чередуют: запись в неё на ходе i + 2 возможна только после барьера хода
// i + 1, а к нему все потоки приходят, уже дочитав ход i.
template <int kWarps, typename Key>
__device__ __forceinline__ Key BlockMin(Key key, Key* partial) {
  static_assert(kWarps <= kWarpSize, "per-warp results are folded by one warp");
  const int lane = threadIdx.x % kWarpSize;
  key = WarpMin(key);
  if (lane == 0) partial[threadIdx.x / kWarpSize] = key;
  __syncthreads();
  return WarpMin(partial[lane < kWarps ? lane : 0]);
}

// Сумма по блоку, верна у потока 0.
template <int kThreads>
__device__ __forceinline__ long long BlockSum(long long value) {
  constexpr int kWarps = kThreads / kWarpSize;
  __shared__ long long partial[kWarps];
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) value += __shfl_down_sync(kFullWarp, value, offset);
  if (threadIdx.x % kWarpSize == 0) partial[threadIdx.x / kWarpSize] = value;
  __syncthreads();
  value = 0;
  for (int w = 0; w < kWarps; ++w) value += partial[w];
  return value;
}

// Постоянная сетка: блок берёт следующую особь из общего счётчика, пока они не кончатся.
__device__ __forceinline__ int ClaimIndividual(int* next_individual) {
  __shared__ int claimed;
  __syncthreads();  // прошлую особь дочитали все: её claimed и буфер минимума можно переписывать
  if (threadIdx.x == 0) claimed = atomicAdd(next_individual, 1);
  __syncthreads();
  return claimed;
}

// Итог спуска особи.
__device__ __forceinline__ void StoreDescent(long long* f, long long objective, unsigned long long* moves,
                                             unsigned long long applied) {
  if (threadIdx.x != 0) return;
  *f = objective;
  atomicAdd(moves, applied);
}

// Ключ хода для минимума по блоку: Δ со смещением в старшей половине, ход в младшей. Улучшающий ход меняет только
// n - 1 пар с участием вершины, так что Δ из [1 - n, -1], и при n < kOffset старшая половина положительна. Меньший
// ключ — меньшая Δ, затем меньшая вершина и кластер: ничьи разрешаются так же, как на CPU. Смещение — константа, а не
// n: так оно не занимает регистр.
template <typename Key>
struct MoveKey {
  static constexpr int kHalfBits = 4 * sizeof(Key);
  static constexpr unsigned kOffset = 1u << (kHalfBits - 1);
  static constexpr Key kNone = ~Key(0);  // больше ключа любого хода
  static_assert(sizeof(Key) < sizeof(unsigned long long) || kMaxVerticesWide < kOffset, "n must fit below the offset");

  __device__ static Key Pack(int delta, unsigned move) { return (Key)((unsigned)delta + kOffset) << kHalfBits | move; }
  __device__ static int Delta(Key key) { return (int)((unsigned)(key >> kHalfBits) - kOffset); }
  __device__ static unsigned Move(Key key) { return (unsigned)(key & ((Key(1) << kHalfBits) - 1)); }
};

// Ход в младшей половине 64-битного ключа: вершина << 2B | откуда << B | куда, по B бит на кластер.
template <int kClusterBits>
struct PackedMove {
  static constexpr unsigned kMask = (1u << kClusterBits) - 1;
  static_assert((unsigned long long)(kMaxVerticesWide - 1) << (2 * kClusterBits) <= 0xffffffffull,
                "packed move must fit 32 bits");

  __device__ static unsigned Pack(int v, int from, int to) {
    return (unsigned)v << (2 * kClusterBits) | (unsigned)from << kClusterBits | (unsigned)to;
  }
  __device__ static int Vertex(unsigned move) { return (int)(move >> (2 * kClusterBits)); }
  __device__ static int From(unsigned move) { return (int)((move >> kClusterBits) & kMask); }
  __device__ static int To(unsigned move) { return (int)(move & kMask); }
};

// Значения шаблонного параметра, для которых собраны экземпляры ядра.
template <int... Values>
constexpr std::array<int, sizeof...(Values)> ValuesOf(std::integer_sequence<int, Values...>) {
  return {Values...};
}

// Вызывает body(std::integral_constant<int, V>{}) для V из Values, равного value: так выбирается экземпляр ядра по
// числу, известному только во время выполнения, без switch по каждому значению.
template <int... Values, typename Body>
void WithConstant(int value, std::integer_sequence<int, Values...>, Body&& body) {
  const bool found = ((value == Values ? (body(std::integral_constant<int, Values>{}), true) : false) || ...);
  if (!found) throw std::logic_error("no kernel instance for " + std::to_string(value));
}

}  // namespace cc::gpu
