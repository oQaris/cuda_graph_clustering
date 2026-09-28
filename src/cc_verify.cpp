// Проверки инкрементальной целевой функции. Формула f = m + sum_c C(n_c,2) - 2W и дельта хода общие с ядрами CUDA,
// так что проверка здесь проверяет и арифметику GPU.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "cc_graph.hpp"
#include "cc_math.hpp"
#include "cc_pbils.hpp"

namespace {

// Временные файлы — туда, что есть на любой платформе: /tmp в Windows нет.
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

// Состояние со случайными метками и пересчитанной G.
cc::State RandomState(const cc::Graph& graph, int k, uint64_t& rng) {
  cc::State state;
  state.Init(static_cast<int>(graph.Size()), k);
  for (int& label : state.labels) label = static_cast<int>(cc::RandBelow(rng, static_cast<unsigned>(k)));
  state.Rebuild(graph);
  return state;
}

// 1. Замкнутая формула целевой функции должна совпадать с попарным подсчётом при любом k.
void TestObjectiveMatchesDefinition() {
  std::printf("objective formula vs pairwise definition\n");
  uint64_t rng = 12345;
  for (const unsigned n : {1u, 2u, 3u, 7u, 40u, 137u}) {
    for (const double density : {0.0, 0.15, 0.5, 0.9, 1.0}) {
      for (const int k : {1, 2, 3, 5, 9}) {
        const cc::Graph graph = cc::Graph::ErdosRenyi(n, density, cc::NextU32(rng));
        const cc::State state = RandomState(graph, k, rng);
        CheckEq(state.f, cc::ObjectiveDirect(graph, state.labels),
                "n=" + std::to_string(n) + " p=" + std::to_string(density) + " k=" + std::to_string(k));
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
      cc::State state = RandomState(graph, k, rng);

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
        const bool g_equal = state.g == fresh.g;
        const bool sizes_equal = state.sizes == fresh.sizes;
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
      cc::State state = RandomState(graph, k, rng);
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
    uint64_t rng = 5;
    cc::State state = RandomState(graph, 3, rng);
    cc::LocalSearch(graph, state);
    CheckEq(state.f, 0, "complete graph K" + std::to_string(n) + " reaches 0");
    CheckEq(cc::CountClustersUsed(state.labels, 3), 1, "K" + std::to_string(n) + " uses one cluster");
  }

  // Граф без рёбер на 4 вершинах при k = 2: лучшее разбиение 2 + 2, его стоимость C(2,2) + C(2,2) = 2.
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
    const std::pair<unsigned, unsigned> edges[] = {{0, 1}, {1, 2}, {0, 2}, {3, 4}, {4, 5}, {3, 5}};
    for (const auto& [from, to] : edges) graph.AddEdge(from, to);
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
  const long long edges = static_cast<long long>(graph.EdgeCount());
  CheckEq(static_cast<long long>(from_matrix.EdgeCount()), edges, "matrix round trip edges");
  CheckEq(static_cast<long long>(from_json.EdgeCount()), edges, "json round trip edges");

  bool same = true;
  for (unsigned i = 0; i < graph.Size() && same; ++i) {
    for (unsigned j = 0; j < graph.Size(); ++j) {
      if (graph.IsJoined(i, j) != from_matrix.IsJoined(i, j) || graph.IsJoined(i, j) != from_json.IsJoined(i, j)) {
        same = false;
        break;
      }
    }
  }
  Check(same, "adjacency preserved by both formats");
  std::remove(matrix_path.c_str());
  std::remove(json_path.c_str());
}

// 6. Решатель должен возвращать разметку, чьё заявленное значение — настоящее, с GWW и без.
void TestSolverReportsTruth() {
  std::printf("solver result is self-consistent\n");
  const cc::Graph graph = cc::Graph::ErdosRenyi(120, 0.5, 4242);
  for (const int k : {2, 3, 6}) {
    for (const double gww : {0.0, 0.25}) {
      cc::PbilsParams params;
      params.k = k;
      params.population = 16;
      params.iterations = 12;
      params.early_stop = 4;
      params.seed = 7;
      params.gww = gww;
      const cc::PbilsResult result = cc::SolveCpu(graph, params);
      const std::string what = "k=" + std::to_string(k) + " gww=" + std::to_string(gww);
      CheckEq(result.objective, cc::ObjectiveDirect(graph, result.labels), "reported objective, " + what);
      Check(result.clusters_used >= 1 && result.clusters_used <= k, "clusters used within bound, " + what);
    }
  }
}

// 7. Увеличение k не должно ухудшать достижимый оптимум: при том же сиде больший бюджет кластеров
//    лишь расширяет пространство поиска.
void TestMoreClustersDoNotHurt() {
  std::printf("edgeless graph: optimum follows the balanced split\n");
  // Для графа без рёбер целевая функция — ровно sum_c C(n_c,2), минимум даёт самое сбалансированное разбиение; с ним и
  // сравниваем в замкнутой форме.
  for (const unsigned n : {6u, 9u, 12u}) {
    for (const int k : {2, 3, 4}) {
      const cc::Graph graph(n);
      uint64_t rng = n * 31 + k;
      cc::State state = RandomState(graph, k, rng);
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

// 8. Пересчёт по словам строк, которым проверяются большие графы, должен совпадать с попарным и на
//    размерах вокруг границы слова, где мешают биты дополнения.
void TestBitwiseObjective() {
  std::printf("bitwise objective vs pairwise definition\n");
  uint64_t rng = 31337;
  for (const unsigned n : {1u, 2u, 31u, 32u, 33u, 63u, 64u, 65u, 200u, 517u}) {
    for (const double density : {0.0, 0.3, 0.8, 1.0}) {
      for (const int k : {1, 2, 3, 7}) {
        const cc::Graph graph = cc::Graph::ErdosRenyi(n, density, cc::NextU32(rng));
        std::vector<int> labels(n);
        for (unsigned v = 0; v < n; ++v) labels[v] = static_cast<int>(cc::RandBelow(rng, static_cast<unsigned>(k)));
        CheckEq(cc::ObjectiveBitwise(graph, labels), cc::ObjectiveDirect(graph, labels),
                "n=" + std::to_string(n) + " p=" + std::to_string(density) + " k=" + std::to_string(k));
      }
    }
  }
}

// Сходство наборов тегов ровно так, как его считает TagsGraphFactory::Chance оригинала: через std::set строк и те же
// формулы в double.
double ChanceLikeBaseline(const std::string& kind, const std::vector<std::string>& a,
                          const std::vector<std::string>& b) {
  const std::set<std::string> set_1(a.begin(), a.end());
  const std::set<std::string> set_2(b.begin(), b.end());
  std::set<std::string> inter;
  std::set_intersection(set_1.begin(), set_1.end(), set_2.begin(), set_2.end(), std::inserter(inter, inter.begin()));
  std::set<std::string> uni;
  std::set_union(set_1.begin(), set_1.end(), set_2.begin(), set_2.end(), std::inserter(uni, uni.begin()));
  const double i = static_cast<double>(inter.size());
  const double s1 = static_cast<double>(set_1.size());
  const double s2 = static_cast<double>(set_2.size());
  if (kind == "jaccard") return i / static_cast<double>(uni.size());
  if (kind == "cosine") return i / std::sqrt(s1 * s2);
  if (kind == "dice") return 2.0 * i / (s1 + s2);
  return i / std::min(s1, s2);
}

// 9. Граф по тегам должен совпадать с попарным построением оригинала: при любой мере и пороге, с пустыми
//    наборами, повторами тегов, экранированием в строках и на выборке объектов.
void TestTagsGraph() {
  std::printf("tags graph vs pairwise construction\n");
  // Теги объекта через пробел: пустые наборы, повторы, кавычка внутри тега.
  std::vector<std::vector<std::string>> objects;
  for (const char* tags :
       {"a", "a b", "b a a", "c", "", "a b c", "d", "a", "b c", "q\"x", "a c d e", "e", "a b", "", "c d", "q\"x a"}) {
    std::istringstream in(tags);
    objects.emplace_back(std::istream_iterator<std::string>(in), std::istream_iterator<std::string>());
  }
  const std::string path = TempPath("cc_verify_tags.json");
  {
    std::ofstream out(path);
    out << "{\n";
    for (size_t i = 0; i < objects.size(); ++i) {
      out << "  \"" << 1000 + i * 7 << "\": [";
      for (size_t t = 0; t < objects[i].size(); ++t) {
        std::string escaped;
        for (const char ch : objects[i][t]) {
          if (ch == '"' || ch == '\\') escaped += '\\';
          escaped += ch;
        }
        out << (t ? ", " : "") << "\"" << escaped << "\"";
      }
      out << "]" << (i + 1 != objects.size() ? "," : "") << "\n";
    }
    out << "}\n";
  }

  const unsigned total = static_cast<unsigned>(objects.size());
  for (const char* kind : {"jaccard", "cosine", "dice", "overlap"}) {
    for (const double threshold : {0.25, 0.5, 0.75, 1.0}) {
      for (const unsigned n : {0u, 9u}) {
        const std::vector<unsigned> chosen = cc::SampleWithoutReplacement(total, n, 42);
        const cc::Graph graph = cc::Graph::LoadTags(path, kind, threshold, n, 42);
        const std::string what = std::string(kind) + " t=" + std::to_string(threshold) + " n=" + std::to_string(n);
        CheckEq(graph.Size(), static_cast<long long>(chosen.size()), what + " size");
        long long edges = 0;
        bool same = true;
        for (unsigned i = 0; i < chosen.size(); ++i) {
          same = same && !graph.IsJoined(i, i);
          for (unsigned j = i + 1; j < chosen.size(); ++j) {
            const bool want = ChanceLikeBaseline(kind, objects[chosen[i]], objects[chosen[j]]) >= threshold;
            same = same && graph.IsJoined(i, j) == want && graph.IsJoined(j, i) == want;
            edges += want;
          }
        }
        Check(same, what + " adjacency");
        CheckEq(static_cast<long long>(graph.EdgeCount()), edges, what + " edges");
      }
    }
  }
  std::remove(path.c_str());
}

// 10. Список рёбер: комментарии, заголовок CSV, разделители, лишние столбцы, повторы, обратные рёбра, петли,
//     разреженные номера и CRLF.
void TestEdgeList() {
  std::printf("edge list loader\n");
  const std::string path = TempPath("cc_verify_edges.txt");
  {
    std::ofstream out(path, std::ios::binary);
    out << "# comment\n% another\nid_1,id_2\n10 20\n20 10\n10,30\r\n30\t40 7.5\n40 40\n\n1000 10 1690000000\n";
  }
  const cc::Graph graph = cc::Graph::LoadEdgeList(path);
  CheckEq(graph.Size(), 5, "edge list: vertices are the distinct ids");
  CheckEq(static_cast<long long>(graph.EdgeCount()), 4, "edge list: duplicates and loops dropped");
  // Номера 10, 20, 30, 40, 1000 -> вершины 0..4.
  const std::set<std::pair<unsigned, unsigned>> want = {{0, 1}, {0, 2}, {2, 3}, {0, 4}};
  bool same = true;
  for (unsigned i = 0; i < graph.Size(); ++i) {
    for (unsigned j = 0; j < graph.Size(); ++j) {
      same = same && graph.IsJoined(i, j) == (want.count({std::min(i, j), std::max(i, j)}) > 0);
    }
  }
  Check(same, "edge list: adjacency");
  std::remove(path.c_str());
}

// 11. GWW: слоты по f, при равных — по номеру; копий — целая часть доли; ответ не зависит от числа потоков.
void TestGoWithTheWinners() {
  std::printf("GWW: slot order, copy count, thread count\n");
  const std::vector<long long> f = {5, 3, 5, 1, 3};
  const std::vector<int> order = cc::RankSlots(f.data(), (int)f.size());
  Check(order == std::vector<int>({3, 1, 4, 0, 2}), "slot order by f, then by slot");
  CheckEq(cc::WinnerCopies(0.1, 128), 12, "copies at gww 0.1, population 128");
  CheckEq(cc::WinnerCopies(cc::kMaxGwwShare, 7), 3, "copies never overlap winners");

  const cc::Graph graph = cc::Graph::ErdosRenyi(150, 0.4, 99);
  cc::PbilsParams params;
  params.k = 3;
  params.population = 24;
  params.iterations = 15;
  params.early_stop = params.iterations;
  params.gww = 0.25;
  params.threads = 1;
  const cc::PbilsResult one = cc::SolveCpu(graph, params);
  params.threads = 4;
  const cc::PbilsResult four = cc::SolveCpu(graph, params);
  CheckEq(four.objective, one.objective, "GWW: objective with 4 threads");
  CheckEq(four.iterations_done, one.iterations_done, "GWW: iterations with 4 threads");
  Check(four.labels == one.labels, "GWW: labels with 4 threads");
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
  TestBitwiseObjective();
  TestTagsGraph();
  TestEdgeList();
  TestGoWithTheWinners();

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
