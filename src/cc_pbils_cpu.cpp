#include "cc_pbils.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "cc_math.hpp"

namespace cc {
namespace {

inline int PopCount(Word x) { return HostPopCount(x); }

double SecondsSince(const std::chrono::steady_clock::time_point& start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// Пул потоков на время одного SolveCpu. Потоки создаются один раз, а не на каждую итерацию: при
// сотне итераций пересоздание съедало бы заметную долю короткого прогона, а ради него и меряем.
//
// Слоты раздаются атомарным счётчиком, а не нарезаются на равные диапазоны: локальный поиск у
// разных особей сходится за разное число ходов, и при статическом разбиении часть ядер простаивала
// бы до конца итерации. Вызывающий поток работает наравне с остальными, поэтому потоков создаётся
// workers - 1.
class WorkerPool {
 public:
  explicit WorkerPool(int workers) {
    for (int i = 1; i < workers; ++i) threads_.emplace_back([this] { Loop(); });
  }

  WorkerPool(const WorkerPool&) = delete;
  WorkerPool& operator=(const WorkerPool&) = delete;

  ~WorkerPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
      ++generation_;
    }
    start_.notify_all();
    for (std::thread& thread : threads_) thread.join();
  }

  // Выполняет body(0), ..., body(count - 1) и возвращается, когда отработали все.
  void Run(int count, const std::function<void(int)>& body) {
    if (threads_.empty()) {
      for (int slot = 0; slot < count; ++slot) body(slot);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      body_ = &body;
      count_ = count;
      next_.store(0, std::memory_order_relaxed);
      busy_ = (int)threads_.size();
      ++generation_;
    }
    start_.notify_all();
    Drain();
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [this] { return busy_ == 0; });
  }

 private:
  void Drain() {
    for (int slot = next_.fetch_add(1, std::memory_order_relaxed); slot < count_;
         slot = next_.fetch_add(1, std::memory_order_relaxed)) {
      (*body_)(slot);
    }
  }

  void Loop() {
    unsigned seen = 0;
    while (true) {
      std::unique_lock<std::mutex> lock(mutex_);
      start_.wait(lock, [&] { return generation_ != seen; });
      seen = generation_;
      if (stop_) return;
      lock.unlock();
      Drain();
      lock.lock();
      if (--busy_ == 0) done_.notify_one();
    }
  }

  std::vector<std::thread> threads_;
  std::mutex mutex_;
  std::condition_variable start_;
  std::condition_variable done_;
  const std::function<void(int)>* body_ = nullptr;
  int count_ = 0;
  std::atomic<int> next_{0};
  unsigned generation_ = 0;
  int busy_ = 0;
  bool stop_ = false;
};

}  // namespace

// Прямой подсчёт по определению, за O(n^2): эталон для проверки быстрых путей.
long long ObjectiveDirect(const Graph& graph, const std::vector<int>& labels) {
  const unsigned n = graph.Size();
  if (labels.size() != n) throw std::runtime_error("labels length != graph size");
  long long distance = 0;
  for (unsigned i = 0; i < n; ++i) {
    for (unsigned j = i + 1; j < n; ++j) {
      const bool same = labels[i] == labels[j];
      const bool joined = graph.IsJoined(i, j);
      if (same != joined) ++distance;
    }
  }
  return distance;
}

int CountClustersUsed(const std::vector<int>& labels, int k) {
  std::vector<char> seen(k, 0);
  for (const int label : labels) {
    if (label >= 0 && label < k) seen[label] = 1;
  }
  int used = 0;
  for (const char s : seen) used += s;
  return used;
}

void State::Init(int vertices, int clusters) {
  n = vertices;
  k = clusters;
  labels.assign(n, 0);
  g.assign((size_t)k * n, 0);
  sizes.assign(k, 0);
  f = 0;
}

// Пересчитывает G, размеры кластеров и f по labels.
void State::Rebuild(const Graph& graph) {
  const unsigned words = graph.WordsPerRow();

  sizes.assign(k, 0);
  for (int v = 0; v < n; ++v) ++sizes[labels[v]];

  std::vector<Word> masks((size_t)k * words, 0u);
  for (int v = 0; v < n; ++v) {
    masks[(size_t)labels[v] * words + v / kWordBits] |= (1u << (v % kWordBits));
  }

  g.assign((size_t)k * n, 0);
  long long intra_twice = 0;
  for (int v = 0; v < n; ++v) {
    const Word* row = graph.Row(v);
    for (int c = 0; c < k; ++c) {
      const Word* mask = masks.data() + (size_t)c * words;
      int acc = 0;
      for (unsigned w = 0; w < words; ++w) acc += PopCount(row[w] & mask[w]);
      g[(size_t)c * n + v] = acc;
    }
    intra_twice += g[(size_t)labels[v] * n + v];
  }

  f = Objective((long long)graph.EdgeCount(), sizes.data(), k, intra_twice / 2);
}

// Индекс g[(size_t)c * n + v] считается на месте намеренно: вынос указателей строк G в отдельный
// массив выглядит ускорением, но добавляет косвенность и на замере оказался ~10% медленнее.
int State::BestMove(int& out_v, int& out_to) const {
  int best = 0;
  out_v = -1;
  out_to = -1;
  for (int v = 0; v < n; ++v) {
    const int from = labels[v];
    const int g_from = g[(size_t)from * n + v];
    const int size_from = sizes[from];
    for (int to = 0; to < k; ++to) {
      if (to == from) continue;
      const int delta = MoveDelta(size_from, sizes[to], g_from, g[(size_t)to * n + v]);
      if (delta < best) {
        best = delta;
        out_v = v;
        out_to = to;
      }
    }
  }
  return best;
}

void State::ApplyMove(const Graph& graph, int v, int to) {
  const int from = labels[v];
  if (from == to) return;

  const int delta = MoveDelta(sizes[from], sizes[to], g[(size_t)from * n + v], g[(size_t)to * n + v]);

  int* g_from = g.data() + (size_t)from * n;
  int* g_to = g.data() + (size_t)to * n;
  const Word* row = graph.Row(v);
  const unsigned words = graph.WordsPerRow();
  for (unsigned w = 0; w < words; ++w) {
    Word bits = row[w];
    while (bits) {
      const unsigned u = w * kWordBits + HostLowestSetBit(bits);
      bits &= bits - 1;
      --g_from[u];
      ++g_to[u];
    }
  }

  --sizes[from];
  ++sizes[to];
  labels[v] = to;
  f += delta;
}

// Наискорейший спуск: пока есть улучшающий ход, применяем его.
long long LocalSearch(const Graph& graph, State& state, long long* accepted_moves) {
  int v = -1;
  int to = -1;
  while (state.BestMove(v, to) < 0) {
    state.ApplyMove(graph, v, to);
    if (accepted_moves) ++*accepted_moves;
  }
  return state.f;
}

void Perturb(const Graph& graph, State& state, double probability, uint64_t seed, int slot,
             int iteration) {
  if (state.k < 2) return;
  const float threshold = (float)probability;
  bool touched = false;
  for (int v = 0; v < state.n; ++v) {
    // Сид вершины: (slot, v) задают поток, iteration — шаг в нём. Смещение 0x5000000 разводит шаги
    // возмущения и селекции, у которой свой базовый сид. То же выражение стоит в KernelPerturb.
    uint64_t rng = StreamSeed(seed, (uint64_t)slot * state.n + v, (uint64_t)iteration + 0x5000000ull);
    if (RandFloat(rng) >= threshold) continue;
    const int shift = 1 + (int)RandBelow(rng, state.k - 1);
    state.labels[v] = (state.labels[v] + shift) % state.k;
    touched = true;
  }
  if (touched) state.Rebuild(graph);
}

int ResolveThreads(const PbilsParams& params) {
  int threads = params.threads;
  if (threads <= 0) threads = (int)std::thread::hardware_concurrency();
  if (threads < 1) threads = 1;  // hardware_concurrency вправе вернуть 0
  if (params.population >= 1 && threads > params.population) threads = params.population;
  return threads;
}

PbilsResult SolveCpu(const Graph& graph, const PbilsParams& params) {
  if (params.k < 1) throw std::runtime_error("k must be >= 1");
  if (params.population < 1) throw std::runtime_error("population must be >= 1");

  const auto started = std::chrono::steady_clock::now();
  const int n = (int)graph.Size();
  const int size = params.population;

  // Базовые сиды трёх независимых источников случайности. Константы и порядок вывода те же, что в
  // SolveGpu, поэтому при одинаковых параметрах оба бэкенда идут по одной траектории.
  const uint64_t init_seed = StreamSeed(params.seed, 0xA11CEull, 0);
  const uint64_t select_seed = StreamSeed(params.seed, 0x5E1EC7ull, 0);
  const uint64_t perturb_seed = StreamSeed(params.seed, 0xBEEFull, 0);

  std::vector<State> population(size);
  std::vector<State> next(size);
  std::vector<long long> moves(size, 0);
  WorkerPool pool(ResolveThreads(params));

  pool.Run(size, [&](int slot) {
    population[slot].Init(n, params.k);
    for (int v = 0; v < n; ++v) {
      uint64_t rng = StreamSeed(init_seed, (uint64_t)slot * n + v, 0);
      population[slot].labels[v] = RandBelow(rng, params.k);
    }
    population[slot].Rebuild(graph);
    next[slot].Init(n, params.k);
  });

  PbilsResult result;
  result.objective = population[0].f;
  result.labels = population[0].labels;
  for (int i = 1; i < size; ++i) {
    if (population[i].f < result.objective) {
      result.objective = population[i].f;
      result.labels = population[i].labels;
    }
  }

  int stall = 0;
  for (int iteration = 0; iteration < params.iterations; ++iteration) {
    // Фаза 1: селекция и локальный поиск. Читается только population, пишется только своя ячейка
    // next, поэтому слоты не пересекаются и синхронизация внутри фазы не нужна.
    pool.Run(size, [&](int slot) {
      uint64_t rng = StreamSeed(select_seed, slot, iteration + 1);
      int chosen = (int)RandBelow(rng, size);
      for (int t = 1; t < params.tournament; ++t) {
        const int candidate = (int)RandBelow(rng, size);
        if (population[candidate].f < population[chosen].f) chosen = candidate;
      }

      State& current = next[slot];
      current = population[chosen];  // присваивание переиспользует буферы, аллокаций тут нет
      long long applied = 0;
      LocalSearch(graph, current, &applied);
      moves[slot] = applied;
    });

    // Рекорд снимается между фазами, пока локальные оптимумы ещё не испорчены возмущением.
    int argmin = 0;
    for (int slot = 1; slot < size; ++slot) {
      if (next[slot].f < next[argmin].f) argmin = slot;
    }
    bool improved = false;
    if (next[argmin].f < result.objective) {
      result.objective = next[argmin].f;
      result.labels = next[argmin].labels;
      improved = true;
    }
    for (int slot = 0; slot < size; ++slot) result.accepted_moves += moves[slot];
    result.local_searches += size;
    result.iterations_done = iteration + 1;
    stall = improved ? 0 : stall + 1;

    if (params.verbose) {
      std::printf("  iter %3d  record %lld  stall %d  %.2fs\n", iteration + 1, result.objective, stall,
                  SecondsSince(started));
    }
    if (stall >= params.early_stop) break;
    if (params.time_limit_sec > 0.0 && SecondsSince(started) >= params.time_limit_sec) break;

    // Фаза 2: возмущение и пересчёт G. Тоже по слотам, и тоже без общих данных.
    if (params.k >= 2 && params.perturbation > 0.0) {
      pool.Run(size, [&](int slot) {
        Perturb(graph, next[slot], params.perturbation, perturb_seed, slot, iteration);
      });
    }
    population.swap(next);
  }

  result.seconds = SecondsSince(started);
  result.clusters_used = CountClustersUsed(result.labels, params.k);
  return result;
}

}  // namespace cc
