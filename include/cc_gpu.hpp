// GPU-бэкенд. Та же сигнатура, что у SolveCpu, поэтому оба ведёт один CLI и измеряет одинаково.
#pragma once

#include "cc_graph.hpp"
#include "cc_pbils.hpp"

namespace cc {

// Наибольшее k, которое ядра держат в разделяемой памяти.
constexpr int kMaxClusters = 64;

PbilsResult SolveGpu(const Graph& graph, const PbilsParams& params);

// NeighborhoodCuda и NeighborhoodWithManyLocalSearchesCuda бейзлайна, только k = 2. Для каждой вершины v первый
// кластер — v и её соседи, второй — остальные. Второй алгоритм улучшает каждое разбиение локальным поиском.
// При равном f выбирается меньшая v; при равных улучшениях — меньшая переносимая вершина.
// population ограничивает число разбиений в одном пакете, ls_kernel выбирает ядро спуска. Остальные настройки
// PBILS, кроме verbose, не используются: оба алгоритма всегда перебирают все вершины.
PbilsResult SolveNeighborhoodGpu(const Graph& graph, const PbilsParams& params);
PbilsResult SolveNeighborhoodWithManyLocalSearchesGpu(const Graph& graph, const PbilsParams& params);

}  // namespace cc
