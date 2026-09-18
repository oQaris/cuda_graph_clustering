// Неориентированный невзвешенный граф в виде битовой матрицы смежности.
//
// Строки дополнены до целого числа 32-битных слов, чтобы пересекать строку с маской кластера и
// считать совпадения через popcount. Это и есть битовая форма произведения A*Z: для невзвешенного
// графа она сводит 32 попарных сравнения к одной инструкции — именно отсюда ускорение
// относительно попарного цикла на CPU.
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

  bool IsJoined(unsigned i, unsigned j) const {
    return (Row(i)[j / kWordBits] >> (j % kWordBits)) & 1u;
  }

  void AddEdge(unsigned i, unsigned j);
  double Density() const;

  // G(n, p) с фиксированным сидом — один и тот же инстанс можно отдать и CPU-референсу, и
  // GPU-решателю.
  static Graph ErdosRenyi(unsigned n, double density, uint64_t seed);

  // Простой текст: первая строка — n, затем n строк из n символов '0'/'1' либо из значений 0/1
  // через пробел.
  static Graph LoadMatrix(const std::string& path);
  void SaveMatrix(const std::string& path) const;

  // Блок "graph": [[0,1,...],...], который бейзлайн (BIGADIL/graph_correlation_clustering) пишет
  // в каждый файл результата. Чтение этого блока позволяет посчитать целевую функцию на тех же
  // инстансах, на которых гонялся бейзлайн.
  static Graph LoadBaselineJson(const std::string& path);
  void SaveBaselineJson(const std::string& path) const;

 private:
  unsigned n_ = 0;
  unsigned words_per_row_ = 0;
  uint64_t edges_ = 0;
  std::vector<Word> bits_;
};

}  // namespace cc
