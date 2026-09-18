// PBILS (population based iterated local search) для нестрогой k-корреляционной кластеризации и
// состояние решения, над которым он работает.
//
// Одна итерация повторяет бейзлайн (BIGADIL/graph_correlation_clustering, класс IPLSAlgorithm),
// чтобы GPU-порт оставался с ним сравним:
//
//   для каждого слота популяции:
//     турнирная селекция из текущей популяции
//     локальный поиск до локального оптимума  -> кандидат в рекорд
//     случайное возмущение этого оптимума      -> член следующей популяции
//   останов — когда рекорд не улучшался early_stop итераций подряд
//
// Начальная популяция случайна и намеренно не оптимизируется локальным поиском сразу — как и в
// бейзлайне.
#pragma once

#include <cstdint>
#include <vector>

#include "cc_graph.hpp"

namespace cc {

struct PbilsParams {
  // Где GPU-бэкенд хранит состояние решения во время локального поиска. Оба ядра реализуют один
  // и тот же наискорейший спуск и дают один ответ; разделяемая память намного быстрее, но
  // ограничена по объёму.
  enum LocalSearchKernel { kLocalSearchAuto = 0, kLocalSearchGlobal = 1, kLocalSearchShared = 2 };

  int k = 2;                     // верхняя граница числа кластеров (нестрого)
  int population = 128;
  int tournament = 5;
  int iterations = 100;
  int early_stop = 6;
  double perturbation = 0.4;     // вероятность перемаркировки вершины
  uint64_t seed = 1;
  double time_limit_sec = 0.0;   // 0 отключает лимит
  bool verbose = false;
  int ls_kernel = kLocalSearchAuto;  // только GPU-бэкенд, на CPU игнорируется
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

// Целевая функция напрямую из определения, по парам: O(n^2). Соответствует GetDistanceToGraph
// бейзлайна и служит эталоном для всех быстрых путей.
long long ObjectiveDirect(const Graph& graph, const std::vector<int>& labels);

int CountClustersUsed(const std::vector<int>& labels, int k);

// Решение вместе с данными, делающими оценку хода O(1): G (= A*Z, по кластерам) и размеры
// кластеров.
struct State {
  int n = 0;
  int k = 0;
  std::vector<int> labels;   // n элементов, значения из [0, k)
  std::vector<int> g;        // k * n элементов, g[c * n + v] = |N(v) в кластере c|
  std::vector<int> sizes;    // k элементов
  long long f = 0;

  void Init(int vertices, int clusters);
  // Пересчитывает g, sizes и f по labels. Битовый параллелизм: строка смежности пересекается с
  // маской каждого кластера и считается через popcount.
  void Rebuild(const Graph& graph);
  // Кандидат наискорейшего спуска. Возвращает дельту (< 0 — улучшение) и записывает ход в
  // out_v / out_to.
  int BestMove(int& out_v, int& out_to) const;
  void ApplyMove(const Graph& graph, int v, int to);
};

// Выполняет строго улучшающие одиночные ходы, пока такие остаются.
long long LocalSearch(const Graph& graph, State& state, long long* accepted_moves = nullptr);

// Перемаркирует каждую вершину с вероятностью probability в равновероятно выбранный другой
// кластер, затем пересчитывает инкрементальные данные.
void Perturb(const Graph& graph, State& state, double probability, uint64_t& rng);

PbilsResult SolveCpu(const Graph& graph, const PbilsParams& params);

}  // namespace cc
