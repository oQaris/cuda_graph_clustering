// Undirected unweighted graph stored as a bit-packed adjacency matrix.
//
// Rows are padded to whole 32-bit words so that a row can be intersected with a
// cluster mask and counted with popcount. That is the bit-parallel form of the
// matrix product A*Z: for an unweighted graph it moves 32 pair comparisons into
// a single instruction, which is where the speed-up over the pairwise CPU loop
// comes from.
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
  const Word* Row(unsigned v) const { return bits_.data() + static_cast<size_t>(v) * words_per_row_; }

  bool IsJoined(unsigned i, unsigned j) const {
    return (Row(i)[j / kWordBits] >> (j % kWordBits)) & 1u;
  }

  void AddEdge(unsigned i, unsigned j);
  double Density() const;

  // G(n, p) with a fixed seed, so the very same instance can be handed to the
  // CPU baseline and to the GPU solver.
  static Graph ErdosRenyi(unsigned n, double density, uint64_t seed);

  // Plain text: first line "n", then n rows of n characters '0'/'1' or of
  // whitespace-separated 0/1 values.
  static Graph LoadMatrix(const std::string& path);
  void SaveMatrix(const std::string& path) const;

  // The "graph": [[0,1,...],...] block that the CPU baseline
  // (BIGADIL/graph_correlation_clustering) writes into every result file. Reading
  // it lets us score the exact instances the baseline was run on.
  static Graph LoadBaselineJson(const std::string& path);
  void SaveBaselineJson(const std::string& path) const;

 private:
  unsigned n_ = 0;
  unsigned words_per_row_ = 0;
  uint64_t edges_ = 0;
  std::vector<Word> bits_;
};

}  // namespace cc
