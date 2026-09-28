#include <algorithm>
#include <stdexcept>
#include <vector>

#include "cc_math.hpp"
#include "cc_parallel.hpp"
#include "cc_pbils.hpp"

namespace cc {

// Прямой подсчёт по определению, за O(n^2): эталон для проверки быстрых путей.
long long ObjectiveDirect(const Graph& graph, const std::vector<int>& labels) {
  const unsigned n = graph.Size();
  if (labels.size() != n) throw std::runtime_error("labels length != graph size");
  long long distance = 0;
  for (unsigned i = 0; i < n; ++i) {
    for (unsigned j = i + 1; j < n; ++j) {
      const bool same = labels[i] == labels[j];
      if (same != graph.IsJoined(i, j)) ++distance;
    }
  }
  return distance;
}

long long ObjectiveBitwise(const Graph& graph, const std::vector<int>& labels) {
  const unsigned n = graph.Size();
  if (labels.size() != n) throw std::runtime_error("labels length != graph size");
  const unsigned words = graph.WordsPerRow();
  int k = 0;
  for (const int label : labels) {
    if (label < 0) throw std::runtime_error("negative label");
    k = std::max(k, label + 1);
  }
  std::vector<Word> masks((size_t)k * words, 0u);
  for (unsigned v = 0; v < n; ++v) masks[(size_t)labels[v] * words + v / kWordBits] |= 1u << (v % kWordBits);

  constexpr unsigned kRowsPerItem = 256;  // крупнее — хуже баланс, мельче — чаще общий счётчик
  WorkerPool pool(HardwareThreads());
  std::vector<long long> sums(pool.Workers(), 0);
  pool.Run((int)((n + kRowsPerItem - 1) / kRowsPerItem), [&](int item, int worker) {
    long long acc = 0;
    for (unsigned v = item * kRowsPerItem; v < std::min(n, (item + 1) * kRowsPerItem); ++v) {
      const Word* row = graph.Row(v);
      const Word* mask = masks.data() + (size_t)labels[v] * words;
      for (unsigned w = 0; w < words; ++w) acc += HostPopCount(row[w] ^ mask[w]);
      acc -= 1;  // бит самой вершины: в маске он стоит, в строке нет
    }
    sums[worker] += acc;
  });
  long long total = 0;
  for (const long long sum : sums) total += sum;
  return total / 2;  // каждая пара посчитана с обеих сторон
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

void State::Rebuild(const Graph& graph) {
  const unsigned words = graph.WordsPerRow();
  sizes.assign(k, 0);
  for (int v = 0; v < n; ++v) ++sizes[labels[v]];

  std::vector<Word> masks((size_t)k * words, 0u);
  for (int v = 0; v < n; ++v) masks[(size_t)labels[v] * words + v / kWordBits] |= 1u << (v % kWordBits);

  g.assign((size_t)k * n, 0);
  long long intra_twice = 0;
  for (int v = 0; v < n; ++v) {
    const Word* row = graph.Row(v);
    for (int c = 0; c < k; ++c) {
      const Word* mask = masks.data() + (size_t)c * words;
      int acc = 0;
      for (unsigned w = 0; w < words; ++w) acc += HostPopCount(row[w] & mask[w]);
      g[(size_t)c * n + v] = acc;
    }
    intra_twice += g[(size_t)labels[v] * n + v];
  }
  f = Objective((long long)graph.EdgeCount(), sizes.data(), k, intra_twice / 2);
}

// Индекс g[(size_t)c * n + v] считается на месте намеренно: указатели строк G в отдельном массиве добавляют
// косвенность, и на замере это ~10% медленнее.
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
  for (unsigned w = 0; w < graph.WordsPerRow(); ++w) {
    for (Word bits = row[w]; bits != 0; bits &= bits - 1) {
      const unsigned u = w * kWordBits + HostLowestSetBit(bits);
      --g_from[u];
      ++g_to[u];
    }
  }
  --sizes[from];
  ++sizes[to];
  labels[v] = to;
  f += delta;
}

long long LocalSearch(const Graph& graph, State& state, long long* accepted_moves) {
  int v = -1;
  int to = -1;
  while (state.BestMove(v, to) < 0) {
    state.ApplyMove(graph, v, to);
    if (accepted_moves) ++*accepted_moves;
  }
  return state.f;
}

void Perturb(const Graph& graph, State& state, double probability, uint64_t seed, int slot, int iteration) {
  if (state.k < 2) return;
  bool touched = false;
  for (int v = 0; v < state.n; ++v) {
    const uint64_t index = (uint64_t)slot * state.n + v;
    const int label = PerturbedLabel(seed, index, iteration, state.labels[v], state.k, (float)probability);
    touched |= label != state.labels[v];
    state.labels[v] = label;
  }
  if (touched) state.Rebuild(graph);
}

int ResolveThreads(const PbilsParams& params) {
  const int threads = params.threads > 0 ? params.threads : HardwareThreads();
  return params.population >= 1 ? std::min(threads, params.population) : threads;
}

int WinnerCopies(double gww, int population) { return (int)(gww * population); }

std::vector<int> RankSlots(const long long* f, int population) {
  std::vector<int> order(population);
  for (int slot = 0; slot < population; ++slot) order[slot] = slot;
  std::sort(order.begin(), order.end(), [f](int a, int b) { return f[a] != f[b] ? f[a] < f[b] : a < b; });
  return order;
}

void CheckParams(const PbilsParams& params) {
  if (params.k < 1) throw std::runtime_error("k must be >= 1");
  if (params.population < 1) throw std::runtime_error("population must be >= 1");
  if (params.gww < 0.0 || params.gww > kMaxGwwShare) throw std::runtime_error("gww must be in [0, 0.5]");
}

PbilsResult SolveCpu(const Graph& graph, const PbilsParams& params) {
  CheckParams(params);
  Progress progress(params);
  const int n = (int)graph.Size();
  const int size = params.population;
  const uint64_t init_stream = StreamSeed(params.seed, kInitStream, 0);
  const uint64_t select_stream = StreamSeed(params.seed, kSelectStream, 0);
  const uint64_t perturb_stream = StreamSeed(params.seed, kPerturbStream, 0);

  std::vector<State> population(size);
  std::vector<State> next(size);
  std::vector<long long> fitness(size);
  std::vector<long long> moves(size, 0);
  WorkerPool pool(ResolveThreads(params));

  pool.Run(size, [&](int slot, int) {
    population[slot].Init(n, params.k);
    for (int v = 0; v < n; ++v) {
      population[slot].labels[v] = InitialLabel(init_stream, (uint64_t)slot * n + v, params.k);
    }
    population[slot].Rebuild(graph);
    next[slot].Init(n, params.k);
  });

  PbilsResult result;
  auto offer_record = [&](const std::vector<State>& states) {
    const auto best = std::min_element(states.begin(), states.end(), [](auto& a, auto& b) { return a.f < b.f; });
    if (!result.labels.empty() && best->f >= result.objective) return false;
    result.objective = best->f;
    result.labels = best->labels;
    return true;
  };
  offer_record(population);

  for (int iteration = 0; iteration < params.iterations; ++iteration) {
    // Селекция и спуск: читается только population, пишется только своя ячейка next.
    for (int slot = 0; slot < size; ++slot) fitness[slot] = population[slot].f;
    pool.Run(size, [&](int slot, int) {
      const int chosen = Tournament(select_stream, slot, iteration, fitness.data(), size, params.tournament);
      next[slot] = population[chosen];  // присваивание переиспользует буферы, аллокаций тут нет
      long long applied = 0;            // счётчик на стеке: общий вектор под каждый ход — ложное разделение кэша
      LocalSearch(graph, next[slot], &applied);
      moves[slot] = applied;
    });

    // Рекорд снимается между фазами, пока локальные оптимумы не испорчены возмущением.
    const bool improved = offer_record(next);
    for (const long long applied : moves) result.accepted_moves += applied;
    if (progress.Next(result, improved)) break;

    // GWW: худшие локальные оптимумы заменяются копиями лучших, пока их не развело возмущение.
    const int copies = WinnerCopies(params.gww, size);
    if (copies > 0) {
      for (int slot = 0; slot < size; ++slot) fitness[slot] = next[slot].f;
      const std::vector<int> order = RankSlots(fitness.data(), size);
      pool.Run(copies, [&](int i, int) { next[order[size - 1 - i]] = next[order[i]]; });
    }

    if (params.k >= 2 && params.perturbation > 0.0) {
      pool.Run(size, [&](int slot, int) {
        Perturb(graph, next[slot], params.perturbation, perturb_stream, slot, iteration);
      });
    }
    population.swap(next);
  }

  result.seconds = progress.Seconds();
  result.clusters_used = CountClustersUsed(result.labels, params.k);
  return result;
}

}  // namespace cc
