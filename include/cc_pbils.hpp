// PBILS (population based iterated local search) для нестрогой k-корреляционной кластеризации и состояние решения,
// над которым он работает. Итерация повторяет IPLSAlgorithm бейзлайна (BIGADIL/graph_correlation_clustering):
//
//   для каждого слота популяции:
//     турнирная селекция из текущей популяции
//     локальный поиск до локального оптимума  -> кандидат в рекорд
//     случайное возмущение этого оптимума      -> член следующей популяции
//   останов — когда рекорд не улучшался early_stop итераций подряд
//
// Начальная популяция случайна и, как в бейзлайне, сразу локальным поиском не оптимизируется. Слоты внутри итерации
// независимы: каждый читает общую популяцию и пишет только свою ячейку следующей. На GPU слот получает блок, на CPU —
// поток из пула; у каждого слота свой поток ГПСЧ, поэтому ответ от числа потоков не зависит.
#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "cc_graph.hpp"

namespace cc {

struct PbilsParams {
  // Какие ядра GPU ведут локальный поиск; все проходят одну траекторию. auto выбирает сам. shared и global — общее
  // ядро с состоянием в разделяемой или глобальной памяти. wide — ядра широкого пути с 32-битной G, которые на графе
  // больше 32 767 вершин включаются сами; явно их берут, чтобы сравнить с узкими на малом графе.
  enum LocalSearchKernel {
    kLocalSearchAuto = 0,
    kLocalSearchGlobal = 1,
    kLocalSearchShared = 2,
    kLocalSearchWide = 3,
  };

  int k = 2;  // верхняя граница числа кластеров (нестрого)
  int population = 128;
  int tournament = 5;
  int iterations = 100;
  int early_stop = 6;
  double perturbation = 0.4;  // вероятность перемаркировки вершины
  int threads = 1;            // только CPU: сколько особей популяции считать сразу
  uint64_t seed = 1;
  double time_limit_sec = 0.0;  // 0 отключает лимит
  bool verbose = false;
  int ls_kernel = kLocalSearchAuto;  // только GPU
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

// Время запуска, застой рекорда, печать прогресса и условия останова — общие для обоих бэкендов.
class Progress {
 public:
  explicit Progress(const PbilsParams& params) : params_(params), started_(std::chrono::steady_clock::now()) {}

  double Seconds() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - started_).count(); }

  // Учитывает законченную итерацию. true — пора остановиться.
  bool Next(PbilsResult& result, bool improved) {
    ++result.iterations_done;
    result.local_searches += params_.population;
    stall_ = improved ? 0 : stall_ + 1;
    if (params_.verbose) {
      std::printf("  iter %3d  record %lld  stall %d  %.2fs\n", result.iterations_done, result.objective, stall_,
                  Seconds());
    }
    return stall_ >= params_.early_stop || (params_.time_limit_sec > 0.0 && Seconds() >= params_.time_limit_sec);
  }

 private:
  const PbilsParams& params_;
  std::chrono::steady_clock::time_point started_;
  int stall_ = 0;
};

// Целевая функция по определению, по парам: O(n^2). Соответствует GetDistanceToGraph бейзлайна и служит эталоном для
// всех быстрых путей.
long long ObjectiveDirect(const Graph& graph, const std::vector<int>& labels);

// То же по словам строк: разногласия вершины со всеми — popcount(строка XOR маска её кластера) без её собственного
// бита. O(n^2/32) на всех ядрах, для графов, где попарный подсчёт шёл бы минутами.
long long ObjectiveBitwise(const Graph& graph, const std::vector<int>& labels);

int CountClustersUsed(const std::vector<int>& labels, int k);

// Решение вместе с данными, делающими оценку хода O(1): G (= A*Z, по кластерам) и размеры кластеров.
struct State {
  int n = 0;
  int k = 0;
  std::vector<int> labels;  // n элементов, значения из [0, k)
  std::vector<int> g;       // k * n элементов, g[c * n + v] = |N(v) в кластере c|
  std::vector<int> sizes;   // k элементов
  long long f = 0;

  void Init(int vertices, int clusters);
  // Пересчитывает g, sizes и f по labels: строка смежности пересекается с маской каждого кластера через popcount.
  void Rebuild(const Graph& graph);
  // Лучший ход наискорейшего спуска: возвращает дельту (< 0 — улучшение) и пишет ход в out_v, out_to.
  int BestMove(int& out_v, int& out_to) const;
  void ApplyMove(const Graph& graph, int v, int to);
};

// Выполняет строго улучшающие одиночные ходы, пока такие остаются.
long long LocalSearch(const Graph& graph, State& state, long long* accepted_moves = nullptr);

// Перемаркирует каждую вершину с вероятностью probability в равновероятный другой кластер и пересчитывает G. Сид
// вершины выводится из (seed, slot, vertex, iteration), как в KernelPerturb, а не из общего состояния: только так
// ответ не зависит от порядка и числа потоков, а CPU и GPU идут одной траекторией.
void Perturb(const Graph& graph, State& state, double probability, uint64_t seed, int slot, int iteration);

// Число рабочих потоков: params.threads <= 0 — по числу ядер, и больше одного потока на особь не нужно.
int ResolveThreads(const PbilsParams& params);

PbilsResult SolveCpu(const Graph& graph, const PbilsParams& params);

}  // namespace cc
