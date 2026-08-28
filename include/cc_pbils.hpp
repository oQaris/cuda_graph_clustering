// PBILS - population based iterated local search - for non-strict
// k-correlation clustering, plus the solution state it operates on.
//
// One iteration, following the CPU baseline (BIGADIL/graph_correlation_clustering,
// class IPLSAlgorithm) so that the GPU port stays comparable to it:
//
//   for every slot of the population:
//     tournament selection from the current population
//     local search down to a local optimum   -> candidate for the record
//     random perturbation of that local optimum -> member of the next population
//   stop when the record has not improved for `early_stop` iterations
//
// The initial population is random and is deliberately not locally optimised,
// again matching the baseline.
#pragma once

#include <cstdint>
#include <vector>

#include "cc_graph.hpp"

namespace cc {

struct PbilsParams {
  // Where the GPU backend keeps the solution state during a local search.
  // Both kernels implement the same steepest descent and return the same
  // answer; shared memory is far faster but bounded by its capacity.
  enum LocalSearchKernel { kLocalSearchAuto = 0, kLocalSearchGlobal = 1, kLocalSearchShared = 2 };

  int k = 2;                     // upper bound on clusters (non-strict)
  int population = 128;
  int tournament = 5;
  int iterations = 100;
  int early_stop = 6;
  double perturbation = 0.4;     // per-vertex relabel probability
  uint64_t seed = 1;
  double time_limit_sec = 0.0;   // 0 disables the limit
  bool verbose = false;
  int ls_kernel = kLocalSearchAuto;  // GPU backend only; ignored on the CPU
};

struct PbilsResult {
  long long objective = -1;
  std::vector<int> labels;
  int iterations_done = 0;
  int clusters_used = 0;
  double seconds = 0.0;
  long long local_searches = 0;
  long long accepted_moves = 0;
};

// The objective straight from its definition, pair by pair: O(n^2). Mirrors the
// baseline's GetDistanceToGraph and serves as ground truth for every fast path.
long long ObjectiveDirect(const Graph& graph, const std::vector<int>& labels);

int CountClustersUsed(const std::vector<int>& labels, int k);

// A solution together with the data that makes a move O(1) to evaluate:
// G (= A*Z, cluster-major) and the cluster sizes.
struct State {
  int n = 0;
  int k = 0;
  std::vector<int> labels;   // n entries, values in [0, k)
  std::vector<int> g;        // k * n entries, g[c * n + v] = |N(v) inside c|
  std::vector<int> sizes;    // k entries
  long long f = 0;

  void Init(int vertices, int clusters);
  // Recomputes g, sizes and f from labels. Bit-parallel: intersects each
  // adjacency row with each cluster mask and counts with popcount.
  void Rebuild(const Graph& graph);
  // Steepest-descent candidate. Returns the delta (< 0 when improving) and
  // writes the move into out_v / out_to.
  int BestMove(int& out_v, int& out_to) const;
  void ApplyMove(const Graph& graph, int v, int to);
};

// Runs strictly improving single-vertex moves until no move improves f.
long long LocalSearch(const Graph& graph, State& state, long long* accepted_moves = nullptr);

// Relabels every vertex with probability `probability` to a uniformly chosen
// different cluster, then rebuilds the incremental data.
void Perturb(const Graph& graph, State& state, double probability, uint64_t& rng);

PbilsResult SolveCpu(const Graph& graph, const PbilsParams& params);

}  // namespace cc
