// Neighborhood: GPU должен совпасть с последовательным перебором, включая метки при ничьих и число ходов.
#include <cstdio>
#include <stdexcept>
#include <string>

#include "cc_gpu.hpp"

namespace {

int g_checks = 0;
int g_failures = 0;

void Check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::printf("  FAIL  %s\n", what.c_str());
  }
}

cc::PbilsResult Reference(const cc::Graph& graph, bool local_search) {
  cc::PbilsResult result;
  cc::State state;
  state.Init((int)graph.Size(), 2);
  for (unsigned v = 0; v < graph.Size(); ++v) {
    for (unsigned u = 0; u < graph.Size(); ++u) state.labels[u] = (u == v || graph.IsJoined(v, u)) ? 0 : 1;
    state.Rebuild(graph);
    if (local_search) cc::LocalSearch(graph, state, &result.accepted_moves);
    if (result.objective < 0 || state.f < result.objective) {
      result.objective = state.f;
      result.labels = state.labels;
    }
  }
  return result;
}

void Compare(const cc::Graph& graph, int batch, int kernel) {
  cc::PbilsParams params;
  params.population = batch;
  params.ls_kernel = kernel;
  for (bool local_search : {false, true}) {
    const auto expected = Reference(graph, local_search);
    const auto got = local_search ? cc::SolveNeighborhoodWithManyLocalSearchesGpu(graph, params)
                                  : cc::SolveNeighborhoodGpu(graph, params);
    const std::string context = "n=" + std::to_string(graph.Size()) + " batch=" + std::to_string(batch) +
                                " kernel=" + std::to_string(kernel) + " ls=" + std::to_string(local_search);
    Check(got.objective == expected.objective, context + " objective");
    Check(got.labels == expected.labels, context + " labels and tie-breaking");
    Check(got.objective == cc::ObjectiveDirect(graph, got.labels), context + " pairwise recount");
    Check(got.accepted_moves == expected.accepted_moves, context + " moves");
    Check(got.local_searches == (local_search ? graph.Size() : 0), context + " searches");
    Check(got.clusters_used == cc::CountClustersUsed(expected.labels, 2), context + " clusters");
  }
}

void TestInvalidInput() {
  auto rejected = [](const cc::Graph& graph, const cc::PbilsParams& params) {
    for (auto solve : {&cc::SolveNeighborhoodGpu, &cc::SolveNeighborhoodWithManyLocalSearchesGpu}) {
      bool threw = false;
      try {
        solve(graph, params);
      } catch (const std::runtime_error&) {
        threw = true;
      }
      Check(threw, "invalid input rejected");
    }
  };
  cc::PbilsParams params;
  rejected(cc::Graph(), params);
  params.k = 3;
  rejected(cc::Graph(1), params);
  params.k = 2;
  params.population = 0;
  rejected(cc::Graph(1), params);
}

}  // namespace

int main() {
  try {
    TestInvalidInput();
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
      std::printf("SKIP: CUDA device unavailable; %d input checks, %d failures\n", g_checks, g_failures);
      return g_failures == 0 ? 77 : 1;
    }
    // Все графы до четырёх вершин: пустые и полные графы дают много равенств, n = 1 допускает пустой кластер.
    for (unsigned n = 1; n <= 4; ++n) {
      const unsigned pairs = n * (n - 1) / 2;
      for (unsigned mask = 0; mask < (1u << pairs); ++mask) {
        cc::Graph graph(n);
        unsigned bit = 0;
        for (unsigned v = 0; v < n; ++v) {
          for (unsigned u = v + 1; u < n; ++u, ++bit) {
            if ((mask >> bit) & 1u) graph.AddEdge(v, u);
          }
        }
        Compare(graph, 3, cc::PbilsParams::kLocalSearchAuto);
      }
    }
    // Границы слова и блока, неполный последний пакет, один слот и пакет больше графа; все ядра спуска.
    for (unsigned n : {31u, 32u, 33u, 127u, 129u, 257u}) {
      for (double density : {0.0, 0.2, 0.5, 1.0}) {
        const auto graph = cc::Graph::ErdosRenyi(n, density, 19 + n);
        for (int kernel : {cc::PbilsParams::kLocalSearchAuto, cc::PbilsParams::kLocalSearchGlobal,
                           cc::PbilsParams::kLocalSearchShared, cc::PbilsParams::kLocalSearchWide}) {
          Compare(graph, 17, kernel);
        }
        Compare(graph, 1, cc::PbilsParams::kLocalSearchAuto);
        Compare(graph, (int)n + 3, cc::PbilsParams::kLocalSearchAuto);
      }
    }
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
  } catch (const std::exception& error) {
    std::printf("FAIL: %s\n", error.what());
    return 1;
  }
}
