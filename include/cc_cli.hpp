// Command line front-end shared by the CPU and the CUDA binaries, so both are
// driven identically and their numbers are directly comparable.
#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "cc_graph.hpp"
#include "cc_pbils.hpp"

namespace cc {
namespace cli {

struct Options {
  // instance
  unsigned n = 500;
  double density = 0.5;
  uint64_t graph_seed = 1;
  std::string graph_path;
  std::string graph_json_path;
  std::string save_graph_path;
  std::string save_graph_json_path;
  // algorithm
  PbilsParams params;
  // experiment
  int runs = 1;
  bool verify = true;
  std::string out_path;
  std::string labels_out_path;
};

inline void PrintUsage(const char* program) {
  std::printf(
      "usage: %s [options]\n"
      "\n"
      "instance\n"
      "  --n N                 generate G(n,p) with N vertices (default 500)\n"
      "  --density P           edge probability (default 0.5)\n"
      "  --graph-seed S        seed of the generator, fixes the instance (default 1)\n"
      "  --graph PATH          load a plain matrix instead of generating\n"
      "  --graph-json PATH     load the \"graph\" block of a baseline result file\n"
      "  --save-graph PATH     write the instance as a plain matrix\n"
      "  --save-graph-json PATH  write the instance in the baseline's JSON layout\n"
      "\n"
      "algorithm\n"
      "  --k K                 upper bound on the number of clusters (default 2)\n"
      "  --pop P               population size (default 128)\n"
      "  --tournament T        tournament size (default 5)\n"
      "  --iters I             iteration cap (default 100)\n"
      "  --early-stop E        stop after E iterations without a record (default 6)\n"
      "  --perturb Q           per-vertex relabel probability (default 0.4)\n"
      "  --seed S              seed of the search (default 1)\n"
      "  --time-limit T        wall clock limit in seconds, 0 disables (default 0)\n"
      "  --ls-kernel WHERE     GPU only: auto (default), shared or global - where the\n"
      "                        solution state lives during a local search\n"
      "\n"
      "experiment\n"
      "  --runs R              independent runs, reports min/avg/max (default 1)\n"
      "  --no-verify           skip the O(n^2) recount of the reported objective\n"
      "  --out PATH            write the result as JSON\n"
      "  --labels-out PATH     write the best clustering as plain labels\n"
      "  --verbose             print per-iteration progress\n",
      program);
}

inline bool NeedsValue(const char* flag, int index, int argc) {
  if (index + 1 < argc) return true;
  std::printf("error: %s needs a value\n", flag);
  return false;
}

inline bool Parse(int argc, char** argv, Options& options) {
  for (int i = 1; i < argc; ++i) {
    const char* flag = argv[i];
    auto value = [&]() { return argv[++i]; };

    if (!std::strcmp(flag, "--help") || !std::strcmp(flag, "-h")) {
      PrintUsage(argv[0]);
      return false;
    } else if (!std::strcmp(flag, "--n")) {
      if (!NeedsValue(flag, i, argc)) return false;
      options.n = static_cast<unsigned>(std::strtoul(value(), nullptr, 10));
    } else if (!std::strcmp(flag, "--density")) {
      if (!NeedsValue(flag, i, argc)) return false;
      options.density = std::strtod(value(), nullptr);
    } else if (!std::strcmp(flag, "--graph-seed")) {
      if (!NeedsValue(flag, i, argc)) return false;
      options.graph_seed = std::strtoull(value(), nullptr, 10);
    } else if (!std::strcmp(flag, "--graph")) {
      if (!NeedsValue(flag, i, argc)) return false;
      options.graph_path = value();
    } else if (!std::strcmp(flag, "--graph-json")) {
      if (!NeedsValue(flag, i, argc)) return false;
      options.graph_json_path = value();
    } else if (!std::strcmp(flag, "--save-graph")) {
      if (!NeedsValue(flag, i, argc)) return false;
      options.save_graph_path = value();
    } else if (!std::strcmp(flag, "--save-graph-json")) {
      if (!NeedsValue(flag, i, argc)) return false;
      options.save_graph_json_path = value();
    } else if (!std::strcmp(flag, "--k")) {
      if (!NeedsValue(flag, i, argc)) return false;
      options.params.k = std::atoi(value());
    } else if (!std::strcmp(flag, "--pop")) {
      if (!NeedsValue(flag, i, argc)) return false;
      options.params.population = std::atoi(value());
    } else if (!std::strcmp(flag, "--tournament")) {
      if (!NeedsValue(flag, i, argc)) return false;
      options.params.tournament = std::atoi(value());
    } else if (!std::strcmp(flag, "--iters")) {
      if (!NeedsValue(flag, i, argc)) return false;
      options.params.iterations = std::atoi(value());
    } else if (!std::strcmp(flag, "--early-stop")) {
      if (!NeedsValue(flag, i, argc)) return false;
      options.params.early_stop = std::atoi(value());
    } else if (!std::strcmp(flag, "--perturb")) {
      if (!NeedsValue(flag, i, argc)) return false;
      options.params.perturbation = std::strtod(value(), nullptr);
    } else if (!std::strcmp(flag, "--seed")) {
      if (!NeedsValue(flag, i, argc)) return false;
      options.params.seed = std::strtoull(value(), nullptr, 10);
    } else if (!std::strcmp(flag, "--time-limit")) {
      if (!NeedsValue(flag, i, argc)) return false;
      options.params.time_limit_sec = std::strtod(value(), nullptr);
    } else if (!std::strcmp(flag, "--ls-kernel")) {
      if (!NeedsValue(flag, i, argc)) return false;
      const char* where = value();
      if (!std::strcmp(where, "auto")) {
        options.params.ls_kernel = PbilsParams::kLocalSearchAuto;
      } else if (!std::strcmp(where, "shared")) {
        options.params.ls_kernel = PbilsParams::kLocalSearchShared;
      } else if (!std::strcmp(where, "global")) {
        options.params.ls_kernel = PbilsParams::kLocalSearchGlobal;
      } else {
        std::printf("error: --ls-kernel expects auto, shared or global\n");
        return false;
      }
    } else if (!std::strcmp(flag, "--runs")) {
      if (!NeedsValue(flag, i, argc)) return false;
      options.runs = std::atoi(value());
    } else if (!std::strcmp(flag, "--no-verify")) {
      options.verify = false;
    } else if (!std::strcmp(flag, "--out")) {
      if (!NeedsValue(flag, i, argc)) return false;
      options.out_path = value();
    } else if (!std::strcmp(flag, "--labels-out")) {
      if (!NeedsValue(flag, i, argc)) return false;
      options.labels_out_path = value();
    } else if (!std::strcmp(flag, "--verbose")) {
      options.params.verbose = true;
    } else {
      std::printf("error: unknown option %s\n", flag);
      PrintUsage(argv[0]);
      return false;
    }
  }
  return true;
}

inline Graph LoadInstance(const Options& options) {
  if (!options.graph_json_path.empty()) return Graph::LoadBaselineJson(options.graph_json_path);
  if (!options.graph_path.empty()) return Graph::LoadMatrix(options.graph_path);
  return Graph::ErdosRenyi(options.n, options.density, options.graph_seed);
}

inline void WriteResultJson(const std::string& path, const Graph& graph,
                            const Options& options, const PbilsResult& best,
                            const char* backend, double avg_objective,
                            long long worst_objective, double avg_seconds) {
  std::ofstream out(path);
  if (!out) {
    std::printf("warning: cannot write %s\n", path.c_str());
    return;
  }
  out << "{\n";
  out << "  \"backend\": \"" << backend << "\",\n";
  out << "  \"size\": " << graph.Size() << ",\n";
  out << "  \"edges\": " << graph.EdgeCount() << ",\n";
  out << "  \"density\": " << graph.Density() << ",\n";
  out << "  \"k\": " << options.params.k << ",\n";
  out << "  \"population\": " << options.params.population << ",\n";
  out << "  \"runs\": " << options.runs << ",\n";
  out << "  \"objective function value\": " << best.objective << ",\n";
  out << "  \"objective average\": " << avg_objective << ",\n";
  out << "  \"objective max\": " << worst_objective << ",\n";
  out << "  \"computation time seconds\": " << best.seconds << ",\n";
  out << "  \"computation time average\": " << avg_seconds << ",\n";
  out << "  \"clusters used\": " << best.clusters_used << ",\n";
  out << "  \"clustering vector\": [";
  for (size_t i = 0; i < best.labels.size(); ++i) {
    out << best.labels[i];
    if (i + 1 != best.labels.size()) out << ",";
  }
  out << "]\n}\n";
}

using SolverFn = PbilsResult (*)(const Graph&, const PbilsParams&);

inline int Main(int argc, char** argv, const char* backend, SolverFn solve) {
  Options options;
  if (!Parse(argc, argv, options)) return 1;

  Graph graph;
  try {
    graph = LoadInstance(options);
  } catch (const std::exception& error) {
    std::printf("error: %s\n", error.what());
    return 1;
  }

  if (!options.save_graph_path.empty()) graph.SaveMatrix(options.save_graph_path);
  if (!options.save_graph_json_path.empty()) graph.SaveBaselineJson(options.save_graph_json_path);

  std::printf("backend      %s\n", backend);
  std::printf("instance     n=%u  edges=%llu  density=%.4f\n", graph.Size(),
              static_cast<unsigned long long>(graph.EdgeCount()), graph.Density());
  std::printf("algorithm    PBILS  k=%d  pop=%d  tournament=%d  iters=%d  early-stop=%d  perturb=%.2f\n",
              options.params.k, options.params.population, options.params.tournament,
              options.params.iterations, options.params.early_stop,
              options.params.perturbation);

  PbilsResult best;
  double objective_sum = 0.0;
  double seconds_sum = 0.0;
  long long worst = 0;

  for (int run = 0; run < options.runs; ++run) {
    PbilsParams params = options.params;
    params.seed = options.params.seed + static_cast<uint64_t>(run);

    PbilsResult result;
    try {
      result = solve(graph, params);
    } catch (const std::exception& error) {
      std::printf("error: %s\n", error.what());
      return 1;
    }

    if (options.verify) {
      const long long truth = ObjectiveDirect(graph, result.labels);
      if (truth != result.objective) {
        std::printf("MISMATCH run %d: reported %lld, recount %lld\n", run, result.objective, truth);
        return 2;
      }
    }

    objective_sum += static_cast<double>(result.objective);
    seconds_sum += result.seconds;
    if (run == 0 || result.objective < best.objective) best = result;
    if (run == 0 || result.objective > worst) worst = result.objective;

    std::printf("run %-3d      f=%-10lld  clusters=%-3d  iters=%-4d  %.3fs\n", run,
                result.objective, result.clusters_used, result.iterations_done, result.seconds);
  }

  const double avg_objective = objective_sum / options.runs;
  const double avg_seconds = seconds_sum / options.runs;
  std::printf("---\n");
  std::printf("objective    min=%lld  avg=%.2f  max=%lld\n", best.objective, avg_objective, worst);
  std::printf("time         best=%.3fs  avg=%.3fs\n", best.seconds, avg_seconds);
  if (options.verify) std::printf("verified     reported value recounted pairwise in O(n^2)\n");

  if (!options.out_path.empty()) {
    WriteResultJson(options.out_path, graph, options, best, backend, avg_objective, worst, avg_seconds);
    std::printf("written      %s\n", options.out_path.c_str());
  }
  if (!options.labels_out_path.empty()) {
    std::ofstream out(options.labels_out_path);
    for (size_t v = 0; v < best.labels.size(); ++v) {
      out << best.labels[v] << (v + 1 == best.labels.size() ? "\n" : " ");
    }
    std::printf("written      %s\n", options.labels_out_path.c_str());
  }
  return 0;
}

}  // namespace cli
}  // namespace cc
