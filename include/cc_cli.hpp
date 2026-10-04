// Общий интерфейс командной строки CPU- и CUDA-бинарников: оба запускаются одинаково, и их числа напрямую сравнимы.
#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "cc_graph.hpp"
#include "cc_pbils.hpp"

namespace cc {
namespace cli {

enum Algorithm { kPbils, kNeighborhood, kNeighborhoodWithManyLocalSearches };

inline const char* AlgorithmName(int algorithm) {
  switch (algorithm) {
    case kNeighborhood: return "NeighborhoodCuda";
    case kNeighborhoodWithManyLocalSearches: return "NeighborhoodWithManyLocalSearchesCuda";
    default: return "PBILS";
  }
}

struct Options {
  // инстанс
  unsigned n = 500;
  double density = 0.5;
  uint64_t graph_seed = 1;
  std::string graph_path;
  std::string graph_json_path;
  std::string edges_path;
  std::string save_graph_path;
  std::string save_graph_json_path;
  std::string tags_path;
  std::string similarity = "jaccard";
  double threshold = 0.5;
  // алгоритм
  PbilsParams params;
  int algorithm = kPbils;
  // эксперимент
  int runs = 1;
  bool verify = true;
  std::string out_path;
  std::string labels_out_path;
};

// Числа разбираются так же нестрого, как atoi и strtod: мусор даёт 0.
inline void ParseValue(const char* text, int& field) { field = std::atoi(text); }
inline void ParseValue(const char* text, unsigned& field) { field = (unsigned)std::strtoul(text, nullptr, 10); }
inline void ParseValue(const char* text, uint64_t& field) { field = (uint64_t)std::strtoull(text, nullptr, 10); }
inline void ParseValue(const char* text, double& field) { field = std::strtod(text, nullptr); }
inline void ParseValue(const char* text, std::string& field) { field = text; }

inline std::string ShowValue(const std::string& value) { return value; }
inline std::string ShowValue(double value) {
  char text[32];
  std::snprintf(text, sizeof(text), "%g", value);
  return text;
}
template <typename T>
std::string ShowValue(T value) {
  return std::to_string(value);
}

// Опция командной строки. value — имя значения в справке, у переключателя nullptr. apply возвращает false, если
// значение не подошло; show даёт значение по умолчанию для справки, пустое не печатается.
struct Flag {
  const char* name;
  const char* value;
  const char* help;
  std::function<bool(const char*)> apply;
  std::function<std::string()> show;
};

template <typename T>
Flag Option(const char* name, const char* value, const char* help, T& field) {
  auto apply = [&field](const char* text) {
    ParseValue(text, field);
    return true;
  };
  return {name, value, help, apply, [&field] { return ShowValue(field); }};
}

inline Flag Switch(const char* name, const char* help, bool& field, bool on) {
  auto apply = [&field, on](const char*) {
    field = on;
    return true;
  };
  return {name, nullptr, help, apply, nullptr};
}

// Значение из списка имён.
inline Flag Choice(const char* name, const char* value, const char* help, int& field,
                   std::vector<std::pair<const char*, int>> choices) {
  auto apply = [&field, choices](const char* text) {
    for (const auto& choice : choices) {
      if (std::strcmp(text, choice.first) == 0) {
        field = choice.second;
        return true;
      }
    }
    return false;
  };
  auto show = [&field, choices] {
    for (const auto& choice : choices) {
      if (field == choice.second) return std::string(choice.first);
    }
    return std::string();
  };
  return {name, value, help, apply, show};
}

struct Section {
  const char* title;
  std::vector<Flag> flags;
};

inline std::vector<Section> Flags(Options& o) {
  PbilsParams& p = o.params;
  return {
      {"инстанс",
       {
           Option("--n", "N", "сгенерировать G(n,p) на N вершинах", o.n),
           Option("--density", "P", "вероятность ребра", o.density),
           Option("--graph-seed", "S", "сид генератора, фиксирует инстанс", o.graph_seed),
           Option("--graph", "PATH", "загрузить матрицу вместо генерации", o.graph_path),
           Option("--graph-json", "PATH", "загрузить блок \"graph\" из файла результата бейзлайна", o.graph_json_path),
           Option("--edges", "PATH", "загрузить список рёбер (SNAP, CSV): номера вершин любые, вершина — их ранг",
                  o.edges_path),
           Option("--save-graph", "PATH", "сохранить инстанс как обычную матрицу", o.save_graph_path),
           Option("--save-graph-json", "PATH", "сохранить инстанс в формате JSON бейзлайна", o.save_graph_json_path),
           Option("--tags", "PATH",
                  "граф по тегам (data/Tags_*.json бейзлайна): n случайных объектов по --graph-seed,\n"
                  "ребро при сходстве тегов не ниже порога; --n 0 берёт все объекты",
                  o.tags_path),
           Option("--similarity", "M", "мера для --tags: jaccard, cosine, dice, overlap", o.similarity),
           Option("--threshold", "T", "порог для --tags", o.threshold),
       }},
      {"алгоритм",
       {
           Choice("--algorithm", "NAME",
                  "pbils, NeighborhoodCuda или NeighborhoodWithManyLocalSearchesCuda;\n"
                  "Neighborhood: только GPU, k = 2, полный перебор без лимита времени",
                  o.algorithm,
                  {{"pbils", kPbils},
                   {"NeighborhoodCuda", kNeighborhood},
                   {"NeighborhoodWithManyLocalSearchesCuda", kNeighborhoodWithManyLocalSearches}}),
           Option("--k", "K", "верхняя граница числа кластеров", p.k),
           Option("--pop", "P", "размер популяции; для Neighborhood — разбиений в пакете", p.population),
           Option("--tournament", "T", "размер турнира", p.tournament),
           Option("--iters", "I", "предел числа итераций", p.iterations),
           Option("--early-stop", "E", "остановиться после E итераций без рекорда", p.early_stop),
           Option("--perturb", "Q", "вероятность перемаркировки вершины", p.perturbation),
           Option("--gww", "R", "GWW: доля худших особей, которую заменяют копии лучших, до 0.5; 0 отключает", p.gww),
           Option("--seed", "S", "сид поиска", p.seed),
           Option("--threads", "T", "только CPU: сколько особей популяции считать сразу, 0 - по числу ядер", p.threads),
           Option("--time-limit", "T", "лимит времени в секундах, 0 отключает", p.time_limit_sec),
           Choice("--ls-kernel", "WHERE",
                  "только GPU: где состояние решения во время локального поиска: shared или global;\n"
                  "auto выбирает сам и при k = 2 берёт отдельное ядро с состоянием в регистрах;\n"
                  "wide - широкий путь, 32-битная G; на графе больше 32 767 вершин включается сам",
                  p.ls_kernel,
                  {{"auto", PbilsParams::kLocalSearchAuto},
                   {"shared", PbilsParams::kLocalSearchShared},
                   {"global", PbilsParams::kLocalSearchGlobal},
                   {"wide", PbilsParams::kLocalSearchWide}}),
       }},
      {"эксперимент",
       {
           Option("--runs", "R", "независимых прогонов, печатает min/avg/max", o.runs),
           Switch("--no-verify", "пропустить пересчёт целевой функции", o.verify, false),
           Option("--out", "PATH", "сохранить результат в JSON", o.out_path),
           Option("--labels-out", "PATH", "сохранить лучшую кластеризацию как метки", o.labels_out_path),
           Switch("--verbose", "печатать прогресс по итерациям", p.verbose, true),
       }},
  };
}

inline void PrintUsage(const char* program) {
  constexpr int kHelpColumn = 24;
  Options defaults;
  std::printf("usage: %s [options]\n", program);
  for (const Section& section : Flags(defaults)) {
    std::printf("\n%s\n", section.title);
    for (const Flag& flag : section.flags) {
      std::string head = std::string("  ") + flag.name + (flag.value ? std::string(" ") + flag.value : "");
      head.resize(std::max<size_t>(head.size() + 1, kHelpColumn), ' ');
      std::string help = flag.help;
      const std::string fallback = flag.show ? flag.show() : "";
      if (!fallback.empty()) help += " (по умолчанию " + fallback + ")";
      const std::string indent = "\n" + std::string(kHelpColumn, ' ');  // продолжения справки — с её колонки
      for (size_t at = help.find('\n'); at != std::string::npos; at = help.find('\n', at + indent.size())) {
        help.replace(at, 1, indent);
      }
      std::printf("%s%s\n", head.c_str(), help.c_str());
    }
  }
}

inline bool Parse(int argc, char** argv, Options& options) {
  const std::vector<Section> sections = Flags(options);
  for (int i = 1; i < argc; ++i) {
    const char* name = argv[i];
    if (!std::strcmp(name, "--help") || !std::strcmp(name, "-h")) {
      PrintUsage(argv[0]);
      return false;
    }
    const Flag* flag = nullptr;
    for (const Section& section : sections) {
      for (const Flag& candidate : section.flags) {
        if (!std::strcmp(name, candidate.name)) flag = &candidate;
      }
    }
    if (flag == nullptr) {
      std::printf("error: unknown option %s\n", name);
      PrintUsage(argv[0]);
      return false;
    }
    if (flag->value != nullptr && i + 1 >= argc) {
      std::printf("error: %s needs a value\n", name);
      return false;
    }
    const char* value = flag->value != nullptr ? argv[++i] : nullptr;
    if (!flag->apply(value)) {
      std::printf("error: %s does not accept %s, see --help\n", name, value);
      return false;
    }
  }
  return true;
}

inline Graph LoadInstance(const Options& options) {
  if (!options.tags_path.empty()) {
    return Graph::LoadTags(options.tags_path, options.similarity, options.threshold, options.n, options.graph_seed);
  }
  if (!options.graph_json_path.empty()) return Graph::LoadBaselineJson(options.graph_json_path);
  if (!options.edges_path.empty()) return Graph::LoadEdgeList(options.edges_path);
  if (!options.graph_path.empty()) return Graph::LoadMatrix(options.graph_path);
  return Graph::ErdosRenyi(options.n, options.density, options.graph_seed);
}

inline void WriteResultJson(const std::string& path, const Graph& graph, const Options& options,
                            const PbilsResult& best, const char* backend, double avg_objective,
                            long long worst_objective, double avg_seconds) {
  std::ofstream out(path);
  if (!out) {
    std::printf("warning: cannot write %s\n", path.c_str());
    return;
  }
  out << "{\n";
  out << "  \"algorithm\": \"" << AlgorithmName(options.algorithm) << "\",\n";
  out << "  \"backend\": \"" << backend << "\",\n";
  out << "  \"size\": " << graph.Size() << ",\n";
  out << "  \"edges\": " << graph.EdgeCount() << ",\n";
  out << "  \"density\": " << graph.Density() << ",\n";
  out << "  \"k\": " << options.params.k << ",\n";
  out << "  \"population\": " << options.params.population << ",\n";
  out << "  \"gww\": " << options.params.gww << ",\n";
  // Число потоков — часть условий замера.
  if (!std::strcmp(backend, "cpu")) out << "  \"threads\": " << ResolveThreads(options.params) << ",\n";
  out << "  \"runs\": " << options.runs << ",\n";
  out << "  \"objective function value\": " << best.objective << ",\n";
  out << "  \"objective average\": " << avg_objective << ",\n";
  out << "  \"objective max\": " << worst_objective << ",\n";
  out << "  \"computation time seconds\": " << best.seconds << ",\n";
  out << "  \"computation time average\": " << avg_seconds << ",\n";
  out << "  \"clusters used\": " << best.clusters_used << ",\n";
  out << "  \"clustering vector\": [";
  for (size_t i = 0; i < best.labels.size(); ++i) out << best.labels[i] << (i + 1 != best.labels.size() ? "," : "");
  out << "]\n}\n";
}

using SolverFn = PbilsResult (*)(const Graph&, const PbilsParams&);

// До этого размера найденное f перепроверяется попарно, по определению; на большем графе попарный подсчёт шёл бы
// минутами, и его заменяет ObjectiveBitwise.
constexpr unsigned kPairwiseVerifyLimit = 32767;

inline int Main(int argc, char** argv, const char* backend, SolverFn solve, SolverFn neighborhood = nullptr,
                SolverFn many_local_searches = nullptr) {
  Options options;
  if (!Parse(argc, argv, options)) return 1;
  if (options.algorithm == kNeighborhood) solve = neighborhood;
  if (options.algorithm == kNeighborhoodWithManyLocalSearches) solve = many_local_searches;
  if (solve == nullptr) {
    std::printf("error: %s is only available in cc_gpu\n", AlgorithmName(options.algorithm));
    return 1;
  }

  Graph graph;
  try {
    graph = LoadInstance(options);
  } catch (const std::exception& error) {
    std::printf("error: %s\n", error.what());
    return 1;
  }

  if (!options.save_graph_path.empty()) graph.SaveMatrix(options.save_graph_path);
  if (!options.save_graph_json_path.empty()) graph.SaveBaselineJson(options.save_graph_json_path);

  const PbilsParams& p = options.params;
  std::printf("backend      %s\n", backend);
  std::printf("instance     n=%u  edges=%llu  density=%.4f\n", graph.Size(), (unsigned long long)graph.EdgeCount(),
              graph.Density());
  if (options.algorithm == kPbils) {
    std::printf("algorithm    PBILS  k=%d  pop=%d  tournament=%d  iters=%d  early-stop=%d  perturb=%.2f  gww=%.2f\n",
                p.k, p.population, p.tournament, p.iterations, p.early_stop, p.perturbation, p.gww);
  } else {
    std::printf("algorithm    %s  k=%d  batch=%d\n", AlgorithmName(options.algorithm), p.k, p.population);
  }
  // Реально используемое число потоков: --threads 0 — по числу ядер, и больше потока на особь не берётся.
  if (!std::strcmp(backend, "cpu")) std::printf("threads      %d\n", ResolveThreads(p));

  const bool pairwise = graph.Size() <= kPairwiseVerifyLimit;
  PbilsResult best;
  double objective_sum = 0.0;
  double seconds_sum = 0.0;
  long long worst = 0;
  for (int run = 0; run < options.runs; ++run) {
    PbilsParams params = p;
    params.seed = p.seed + (uint64_t)run;

    PbilsResult result;
    try {
      result = solve(graph, params);
    } catch (const std::exception& error) {
      std::printf("error: %s\n", error.what());
      return 1;
    }

    if (options.verify) {
      const long long truth = pairwise ? ObjectiveDirect(graph, result.labels) : ObjectiveBitwise(graph, result.labels);
      if (truth != result.objective) {
        std::printf("MISMATCH run %d: reported %lld, recount %lld\n", run, result.objective, truth);
        return 2;
      }
    }

    objective_sum += (double)result.objective;
    seconds_sum += result.seconds;
    if (run == 0 || result.objective < best.objective) best = result;
    if (run == 0 || result.objective > worst) worst = result.objective;

    std::printf("run %-3d      f=%-10lld  clusters=%-3d  iters=%-4d  %.3fs\n", run, result.objective,
                result.clusters_used, result.iterations_done, result.seconds);
    if (p.verbose && result.local_searches > 0) {
      std::printf("moves        %lld accepted, %.0f per local search\n", result.accepted_moves,
                  (double)result.accepted_moves / (double)result.local_searches);
    }
  }

  const double avg_objective = objective_sum / options.runs;
  const double avg_seconds = seconds_sum / options.runs;
  std::printf("---\n");
  std::printf("objective    min=%lld  avg=%.2f  max=%lld\n", best.objective, avg_objective, worst);
  std::printf("time         best=%.3fs  avg=%.3fs\n", best.seconds, avg_seconds);
  if (options.verify) {
    std::printf(pairwise ? "verified     reported value recounted pairwise in O(n^2)\n"
                         : "verified     reported value recounted over bit rows in O(n^2/32)\n");
  }

  if (!options.out_path.empty()) {
    WriteResultJson(options.out_path, graph, options, best, backend, avg_objective, worst, avg_seconds);
    std::printf("written      %s\n", options.out_path.c_str());
  }
  if (!options.labels_out_path.empty()) {
    std::ofstream out(options.labels_out_path);
    for (size_t v = 0; v < best.labels.size(); ++v) out << best.labels[v] << (v + 1 == best.labels.size() ? "\n" : " ");
    std::printf("written      %s\n", options.labels_out_path.c_str());
  }
  return 0;
}

}  // namespace cli
}  // namespace cc
