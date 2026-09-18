// GPU-бэкенд. Та же сигнатура, что у SolveCpu, поэтому оба ведёт один CLI и измеряет одинаково.
#pragma once

#include "cc_graph.hpp"
#include "cc_pbils.hpp"

namespace cc {

// Наибольшее k, которое ядра держат в разделяемой памяти.
constexpr int kMaxClusters = 64;

PbilsResult SolveGpu(const Graph& graph, const PbilsParams& params);

}  // namespace cc
