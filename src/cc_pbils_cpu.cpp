#include "cc_pbils.hpp"

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <utility>

#include "cc_math.hpp"

namespace cc {
namespace {

inline int PopCount(Word x) { return HostPopCount(x); }

double SecondsSince(const std::chrono::steady_clock::time_point& start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

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
  g.assign(static_cast<size_t>(k) * n, 0);
  sizes.assign(k, 0);
  f = 0;
}

void State::Rebuild(const Graph& graph) {
  const unsigned words = graph.WordsPerRow();

  sizes.assign(k, 0);
  for (int v = 0; v < n; ++v) ++sizes[labels[v]];

  std::vector<Word> masks(static_cast<size_t>(k) * words, 0u);
  for (int v = 0; v < n; ++v) {
    masks[static_cast<size_t>(labels[v]) * words + v / kWordBits] |= (1u << (v % kWordBits));
  }

  g.assign(static_cast<size_t>(k) * n, 0);
  long long intra_twice = 0;
  for (int v = 0; v < n; ++v) {
    const Word* row = graph.Row(v);
    for (int c = 0; c < k; ++c) {
      const Word* mask = masks.data() + static_cast<size_t>(c) * words;
      int acc = 0;
      for (unsigned w = 0; w < words; ++w) acc += PopCount(row[w] & mask[w]);
      g[static_cast<size_t>(c) * n + v] = acc;
    }
    intra_twice += g[static_cast<size_t>(labels[v]) * n + v];
  }

  f = Objective(static_cast<long long>(graph.EdgeCount()), sizes.data(), k, intra_twice / 2);
}

int State::BestMove(int& out_v, int& out_to) const {
  int best = 0;
  out_v = -1;
  out_to = -1;
  for (int v = 0; v < n; ++v) {
    const int from = labels[v];
    const int g_from = g[static_cast<size_t>(from) * n + v];
    const int size_from = sizes[from];
    for (int to = 0; to < k; ++to) {
      if (to == from) continue;
      const int delta = MoveDelta(size_from, sizes[to], g_from, g[static_cast<size_t>(to) * n + v]);
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

  const int delta = MoveDelta(sizes[from], sizes[to],
                              g[static_cast<size_t>(from) * n + v],
                              g[static_cast<size_t>(to) * n + v]);

  int* g_from = g.data() + static_cast<size_t>(from) * n;
  int* g_to = g.data() + static_cast<size_t>(to) * n;
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

long long LocalSearch(const Graph& graph, State& state, long long* accepted_moves) {
  int v = -1;
  int to = -1;
  while (state.BestMove(v, to) < 0) {
    state.ApplyMove(graph, v, to);
    if (accepted_moves) ++*accepted_moves;
  }
  return state.f;
}

void Perturb(const Graph& graph, State& state, double probability, uint64_t& rng) {
  if (state.k < 2) return;
  const float threshold = static_cast<float>(probability);
  bool touched = false;
  for (int v = 0; v < state.n; ++v) {
    if (RandFloat(rng) >= threshold) continue;
    const int shift = 1 + static_cast<int>(RandBelow(rng, static_cast<unsigned>(state.k - 1)));
    state.labels[v] = (state.labels[v] + shift) % state.k;
    touched = true;
  }
  if (touched) state.Rebuild(graph);
}

PbilsResult SolveCpu(const Graph& graph, const PbilsParams& params) {
  if (params.k < 1) throw std::runtime_error("k must be >= 1");
  if (params.population < 1) throw std::runtime_error("population must be >= 1");

  const auto started = std::chrono::steady_clock::now();
  const int n = static_cast<int>(graph.Size());
  const int size = params.population;

  std::vector<State> population(size);
  std::vector<State> next(size);
  uint64_t rng = StreamSeed(params.seed, 0x9110E5ull, 0);

  for (int i = 0; i < size; ++i) {
    population[i].Init(n, params.k);
    for (int v = 0; v < n; ++v) {
      population[i].labels[v] = static_cast<int>(RandBelow(rng, static_cast<unsigned>(params.k)));
    }
    population[i].Rebuild(graph);
    next[i].Init(n, params.k);
  }

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
    bool improved = false;
    for (int slot = 0; slot < size; ++slot) {
      int chosen = static_cast<int>(RandBelow(rng, static_cast<unsigned>(size)));
      for (int t = 1; t < params.tournament; ++t) {
        const int candidate = static_cast<int>(RandBelow(rng, static_cast<unsigned>(size)));
        if (population[candidate].f < population[chosen].f) chosen = candidate;
      }

      State current = population[chosen];
      LocalSearch(graph, current, &result.accepted_moves);
      ++result.local_searches;

      if (current.f < result.objective) {
        result.objective = current.f;
        result.labels = current.labels;
        improved = true;
      }

      Perturb(graph, current, params.perturbation, rng);
      next[slot] = std::move(current);
    }
    population.swap(next);

    result.iterations_done = iteration + 1;
    stall = improved ? 0 : stall + 1;

    if (params.verbose) {
      std::printf("  iter %3d  record %lld  stall %d  %.2fs\n", iteration + 1,
                  result.objective, stall, SecondsSince(started));
    }
    if (stall >= params.early_stop) break;
    if (params.time_limit_sec > 0.0 && SecondsSince(started) >= params.time_limit_sec) break;
  }

  result.seconds = SecondsSince(started);
  result.clusters_used = CountClustersUsed(result.labels, params.k);
  return result;
}

}  // namespace cc
