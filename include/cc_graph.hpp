// Неориентированный невзвешенный граф в виде битовой матрицы смежности. Строки дополнены до целого числа 32-битных
// слов, чтобы пересекать строку с маской кластера через popcount: это битовая форма произведения A*Z, которая сводит
// 32 попарных сравнения к одной инструкции.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cc {

using Word = uint32_t;
constexpr unsigned kWordBits = 32;

class Graph {
 public:
  Graph() = default;
  explicit Graph(unsigned n);

  unsigned Size() const { return n_; }
  unsigned WordsPerRow() const { return words_per_row_; }
  uint64_t EdgeCount() const { return edges_; }
  const std::vector<Word>& Bits() const { return bits_; }
  const Word* Row(unsigned v) const { return bits_.data() + (size_t)v * words_per_row_; }

  bool IsJoined(unsigned i, unsigned j) const { return (Row(i)[j / kWordBits] >> (j % kWordBits)) & 1u; }

  void AddEdge(unsigned i, unsigned j);
  double Density() const;

  // G(n, p) по сиду: один и тот же инстанс можно отдать и CPU, и GPU.
  static Graph ErdosRenyi(unsigned n, double density, uint64_t seed);

  // Текст: первая строка — n, затем n строк из n символов '0'/'1' либо значений 0/1 через пробел.
  static Graph LoadMatrix(const std::string& path);
  void SaveMatrix(const std::string& path) const;

  // Блок "graph": [[0,1,...],...], который бейзлайн (BIGADIL/graph_correlation_clustering) пишет в каждый файл
  // результата: так целевую функцию можно посчитать на тех же инстансах, на которых гонялся бейзлайн.
  static Graph LoadBaselineJson(const std::string& path);
  void SaveBaselineJson(const std::string& path) const;

  // Список рёбер, как у SNAP и в CSV: строка — два номера вершин через пробел, табуляцию или запятую, остальное в
  // строке (вес, время) не читается; строки с '#', '%' и не с числа пропускаются. Направление, петли и повторы
  // теряются.
  static Graph LoadEdgeList(const std::string& path);

  // Граф по тегам, как его строит TagsGraphFactory бейзлайна из data/Tags_*.json: объект -> список тегов, ребро между
  // объектами, если мера сходства их наборов тегов (jaccard, cosine, dice или overlap) не ниже порога. Вершины —
  // SampleWithoutReplacement(объектов, n, seed).
  static Graph LoadTags(const std::string& path, const std::string& similarity, double threshold, unsigned n,
                        uint64_t seed);

 private:
  unsigned n_ = 0;
  unsigned words_per_row_ = 0;
  uint64_t edges_ = 0;
  std::vector<Word> bits_;
};

// n случайных номеров из [0, total) без повторов по сиду; n = 0 или n >= total — все по порядку.
std::vector<unsigned> SampleWithoutReplacement(unsigned total, unsigned n, uint64_t seed);

}  // namespace cc
