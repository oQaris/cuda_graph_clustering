// CUDA backend. Same signature as SolveCpu, so both are driven by one CLI and
// measured the same way.
#pragma once

#include "cc_graph.hpp"
#include "cc_pbils.hpp"

namespace cc {

// Largest k the kernels keep in shared memory.
constexpr int kMaxClusters = 64;

PbilsResult SolveGpu(const Graph& graph, const PbilsParams& params);

}  // namespace cc
