// Проверки корректности инкрементальной целевой функции. Формула
//   f = m + sum_c C(n_c,2) - 2W
// и дельта хода общие с ядрами CUDA, поэтому доказательство здесь доказывает
// и арифметику, на которую опирается GPU.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "cc_graph.hpp"
#include "cc_math.hpp"
#include "cc_pbils.hpp"

namespace {

// Временные файлы должны попадать туда, что существует на любой платформе:
// /tmp нет в Windows, поэтому используем собственные переменные окружения.
std::string TempPath(const char* name) {
  for (const char* variable : {"TMPDIR", "TEMP", "TMP"}) {
    if (const char* dir = std::getenv(variable)) {
      std::string path = dir;
      if (!path.empty() && path.back() != '/' && path.back() != '\\') path += '/';
      return path + name;
    }
  }
  return std::string("./") + name;
}

int g_failures = 0;
int g_checks = 0;

void Check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::printf("  FAIL  %s\n", what.c_str());
  }
}

void CheckEq(long long got, long long want, const std::string& what) {
  ++g_checks;
  if (got != want) {
    ++g_failures;
    std::printf("  FAIL  %s: got %lld, want %lld\n", what.c_str(), got, want);
  }
}

// 1. Замкнутая формула целевой функции должна совпадать с попарным подсчётом при любом k.
void TestObjectiveMatchesDefinition() {
  std::printf("objective formula vs pairwise definition\n");
  uint64_t rng = 12345;
  for (const unsigned n : {1u, 2u, 3u, 7u, 40u, 137u}) {
    for (const double density : {0.0, 0.15, 0.5, 0.9, 1.0}) {
      for (const int k : {1, 2, 3, 5, 9}) {
        const cc::Graph graph = cc::Graph::ErdosRenyi(n, density, cc::NextU32(rng));
        cc::State state;
        state.Init(static_cast<int>(n), k);
        for (unsigned v = 0; v < n; ++v) {
          state.labels[v] = static_cast<int>(cc::RandBelow(rng, static_cast<unsigned>(k)));
        }
        state.Rebuild(graph);
        CheckEq(state.f, cc::ObjectiveDirect(graph, state.labels),
                "n=" + std::to_string(n) + " p=" + std::to_string(density) +
                    " k=" + std::to_string(k));
      }
    }
  }
}

// 2. G, sizes и f должны оставаться точными после длинной цепочки инкрементальных ходов.
void TestIncrementalMovesStayExact() {
  std::printf("incremental move updates stay exact\n");
  uint64_t rng = 999;
  for (const unsigned n : {2u, 5u, 31u, 96u}) {
    for (const int k : {2, 3, 7}) {
      const cc::Graph graph = cc::Graph::ErdosRenyi(n, 0.45, cc::NextU32(rng));
      cc::State state;
      state.Init(static_cast<int>(n), k);
      for (unsigned v = 0; v < n; ++v) {
        state.labels[v] = static_cast<int>(cc::RandBelow(rng, static_cast<unsigned>(k)));
      }
      state.Rebuild(graph);

      for (int step = 0; step < 200; ++step) {
        const int v = static_cast<int>(cc::RandBelow(rng, n));
        const int to = static_cast<int>(cc::RandBelow(rng, static_cast<unsigned>(k)));
        if (to == state.labels[v]) continue;
        state.ApplyMove(graph, v, to);

        cc::State fresh;
        fresh.Init(static_cast<int>(n), k);
        fresh.labels = state.labels;
        fresh.Rebuild(graph);

        CheckEq(state.f, fresh.f, "f after move");
        CheckEq(state.f, cc::ObjectiveDirect(graph, state.labels), "f vs definition");
        bool g_equal = state.g == fresh.g;
        bool sizes_equal = state.sizes == fresh.sizes;
        Check(g_equal, "G after move (n=" + std::to_string(n) + " k=" + std::to_string(k) + ")");
        Check(sizes_equal, "sizes after move");
        if (!g_equal || !sizes_equal) return;
      }
    }
  }
}

// 3. Локальный поиск должен строго уменьшать f и останавливаться в настоящем локальном оптимуме.
void TestLocalSearchDescends() {
  std::printf("local search descends to a local optimum\n");
  uint64_t rng = 777;
  for (const unsigned n : {4u, 23u, 80u, 210u}) {
    for (const int k : {2, 3, 4}) {
      const cc::Graph graph = cc::Graph::ErdosRenyi(n, 0.35, cc::NextU32(rng));
      cc::State state;
      state.Init(static_cast<int>(n), k);
      for (unsigned v = 0; v < n; ++v) {
        state.labels[v] = static_cast<int>(cc::RandBelow(rng, static_cast<unsigned>(k)));
      }
      state.Rebuild(graph);
      const long long before = state.f;

      long long moves = 0;
      cc::LocalSearch(graph, state, &moves);

      Check(state.f <= before, "f did not increase");
      CheckEq(state.f, cc::ObjectiveDirect(graph, state.labels), "local optimum vs definition");
      int v = -1, to = -1;
      Check(state.BestMove(v, to) >= 0, "no improving move remains");
    }
  }
}

// 4. Инстансы, оптимум которых известен заранее, вручную.
void TestKnownOptima() {
  std::printf("hand-checked instances\n");

  // Полный граф — уже один кластер: разногласий нет.
  for (const unsigned n : {2u, 6u, 33u}) {
    const cc::Graph graph = cc::Graph::ErdosRenyi(n, 1.0, 1);
    cc::State state;
    state.Init(static_cast<int>(n), 3);
    uint64_t rng = 5;
    for (unsigned v = 0; v < n; ++v) {
      state.labels[v] = static_cast<int>(cc::RandBelow(rng, 3));
    }
    state.Rebuild(graph);
    cc::LocalSearch(graph, state);
    CheckEq(state.f, 0, "complete graph K" + std::to_string(n) + " reaches 0");
    CheckEq(cc::CountClustersUsed(state.labels, 3), 1, "K" + std::to_string(n) + " uses one cluster");
  }

  // Граф без рёбер на 4 вершинах при k = 2: лучшее разбиение 2 + 2, его
  // стоимость C(2,2) + C(2,2) = 2.
  {
    const cc::Graph graph(4);
    cc::State state;
    state.Init(4, 2);
    state.labels = {0, 0, 0, 0};
    state.Rebuild(graph);
    CheckEq(state.f, 6, "empty graph, all together");
    cc::LocalSearch(graph, state);
    CheckEq(state.f, 2, "empty graph, k=2 optimum");
  }

  // Два непересекающихся треугольника: k=2 разделяет их бесплатно.
  {
    cc::Graph graph(6);
    graph.AddEdge(0, 1); graph.AddEdge(1, 2); graph.AddEdge(0, 2);
    graph.AddEdge(3, 4); graph.AddEdge(4, 5); graph.AddEdge(3, 5);
    cc::State state;
    state.Init(6, 2);
    state.labels = {0, 1, 0, 1, 0, 1};
    state.Rebuild(graph);
    cc::LocalSearch(graph, state);
    CheckEq(state.f, 0, "two triangles, k=2");
  }

  // Одна вершина: разногласий не бывает.
  {
    const cc::Graph graph(1);
    cc::State state;
    state.Init(1, 2);
    state.labels = {0};
    state.Rebuild(graph);
    CheckEq(state.f, 0, "n=1");
  }
}

// 5. Ввод-вывод графа должен быть обратим, включая формат JSON бейзлайна.
void TestGraphIo() {
  std::printf("graph IO round trip\n");
  const cc::Graph graph = cc::Graph::ErdosRenyi(37, 0.4, 2024);

  const std::string matrix_path = TempPath("cc_verify_matrix.txt");
  const std::string json_path = TempPath("cc_verify_graph.json");
  graph.SaveMatrix(matrix_path);
  graph.SaveBaselineJson(json_path);

  const cc::Graph from_matrix = cc::Graph::LoadMatrix(matrix_path);
  const cc::Graph from_json = cc::Graph::LoadBaselineJson(json_path);

  CheckEq(from_matrix.Size(), graph.Size(), "matrix round trip size");
  CheckEq(from_json.Size(), graph.Size(), "json round trip size");
  CheckEq(static_cast<long long>(from_matrix.EdgeCount()),
          static_cast<long long>(graph.EdgeCount()), "matrix round trip edges");
  CheckEq(static_cast<long long>(from_json.EdgeCount()),
          static_cast<long long>(graph.EdgeCount()), "json round trip edges");

  bool same = true;
  for (unsigned i = 0; i < graph.Size() && same; ++i) {
    for (unsigned j = 0; j < graph.Size(); ++j) {
      if (graph.IsJoined(i, j) != from_matrix.IsJoined(i, j) ||
          graph.IsJoined(i, j) != from_json.IsJoined(i, j)) {
        same = false;
        break;
      }
    }
  }
  Check(same, "adjacency preserved by both formats");
  std::remove(matrix_path.c_str());
  std::remove(json_path.c_str());
}

// 6. Решатель должен возвращать разметку, чьё заявленное значение — настоящее.
void TestSolverReportsTruth() {
  std::printf("solver result is self-consistent\n");
  const cc::Graph graph = cc::Graph::ErdosRenyi(120, 0.5, 4242);
  for (const int k : {2, 3, 6}) {
    cc::PbilsParams params;
    params.k = k;
    params.population = 16;
    params.iterations = 12;
    params.early_stop = 4;
    params.seed = 7;
    const cc::PbilsResult result = cc::SolveCpu(graph, params);
    CheckEq(result.objective, cc::ObjectiveDirect(graph, result.labels),
            "reported objective, k=" + std::to_string(k));
    Check(result.clusters_used >= 1 && result.clusters_used <= k, "clusters used within bound");
  }
}

// 7. Увеличение k не должно ухудшать достижимый оптимум: при том же сиде больший бюджет кластеров
//    лишь расширяет пространство поиска.
void TestMoreClustersDoNotHurt() {
  std::printf("edgeless graph: optimum follows the balanced split\n");
  // Для графа без рёбер целевая функция — ровно sum_c C(n_c,2), минимум даёт самое сбалансированное
  // разбиение; с ним и сравниваем в замкнутой форме.
  for (const unsigned n : {6u, 9u, 12u}) {
    for (const int k : {2, 3, 4}) {
      const cc::Graph graph(n);
      cc::State state;
      state.Init(static_cast<int>(n), k);
      uint64_t rng = n * 31 + k;
      for (unsigned v = 0; v < n; ++v) {
        state.labels[v] = static_cast<int>(cc::RandBelow(rng, static_cast<unsigned>(k)));
      }
      state.Rebuild(graph);
      cc::LocalSearch(graph, state);

      const int base = static_cast<int>(n) / k;
      const int remainder = static_cast<int>(n) % k;
      long long want = 0;
      for (int c = 0; c < k; ++c) {
        const long long s = base + (c < remainder ? 1 : 0);
        want += s * (s - 1) / 2;
      }
      CheckEq(state.f, want, "n=" + std::to_string(n) + " k=" + std::to_string(k));
    }
  }
}

}  // namespace

int main() {
  TestObjectiveMatchesDefinition();
  TestIncrementalMovesStayExact();
  TestLocalSearchDescends();
  TestKnownOptima();
  TestGraphIo();
  TestSolverReportsTruth();
  TestMoreClustersDoNotHurt();

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
