#include "cc_graph.hpp"

#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "cc_math.hpp"

namespace cc {

Graph::Graph(unsigned n) : n_(n) {
  words_per_row_ = (n + kWordBits - 1) / kWordBits;
  if (words_per_row_ == 0) words_per_row_ = 1;
  bits_.assign((size_t)n_ * words_per_row_, 0u);
}

void Graph::AddEdge(unsigned i, unsigned j) {
  if (i == j) return;
  if (IsJoined(i, j)) return;
  bits_[(size_t)i * words_per_row_ + j / kWordBits] |= (1u << (j % kWordBits));
  bits_[(size_t)j * words_per_row_ + i / kWordBits] |= (1u << (i % kWordBits));
  ++edges_;
}

double Graph::Density() const {
  const double pairs = (double)n_ * (n_ - 1) / 2.0;
  return pairs > 0 ? (double)edges_ / pairs : 0.0;
}

Graph Graph::ErdosRenyi(unsigned n, double density, uint64_t seed) {
  Graph g(n);
  uint64_t state = StreamSeed(seed, 0xE4D05Full, 0);
  for (unsigned i = 0; i < n; ++i) {
    for (unsigned j = 0; j < i; ++j) {
      if (RandFloat(state) < density) g.AddEdge(i, j);
    }
  }
  return g;
}

Graph Graph::LoadMatrix(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open graph file: " + path);
  unsigned n = 0;
  in >> n;
  if (!in || n == 0) throw std::runtime_error("bad graph header in " + path);
  Graph g(n);
  for (unsigned i = 0; i < n; ++i) {
    for (unsigned j = 0; j < n; ++j) {
      int value = 0;
      // Принимает и раскладку "0 1 0", и "010".
      int ch = in.get();
      while (ch != EOF && std::isspace(ch)) ch = in.get();
      if (ch == EOF) throw std::runtime_error("truncated matrix in " + path);
      value = (ch == '1') ? 1 : 0;
      if (value && j > i) g.AddEdge(i, j);
    }
  }
  return g;
}

void Graph::SaveMatrix(const std::string& path) const {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write graph file: " + path);
  out << n_ << "\n";
  for (unsigned i = 0; i < n_; ++i) {
    std::string row(n_, '0');
    for (unsigned j = 0; j < n_; ++j) {
      if (IsJoined(i, j)) row[j] = '1';
    }
    out << row << "\n";
  }
}

// Ищет ключ "graph" и читает следующую за ним матрицу 0/1 в скобках. Намеренно сканер, а не
// JSON-парсер: файлы результатов бейзлайна несут всю матрицу вперемешку с посторонними полями, а
// нужен только этот блок.
Graph Graph::LoadBaselineJson(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open json file: " + path);
  std::stringstream buffer;
  buffer << in.rdbuf();
  const std::string text = buffer.str();

  const size_t key = text.find("\"graph\"");
  if (key == std::string::npos) throw std::runtime_error("no \"graph\" key in " + path);
  size_t pos = text.find('[', key);
  if (pos == std::string::npos) throw std::runtime_error("malformed \"graph\" block in " + path);

  std::vector<std::vector<int>> rows;
  int depth = 0;
  std::vector<int> current;
  for (; pos < text.size(); ++pos) {
    const char ch = text[pos];
    if (ch == '[') {
      ++depth;
      if (depth == 2) current.clear();
    } else if (ch == ']') {
      if (depth == 2) rows.push_back(current);
      --depth;
      if (depth == 0) break;
    } else if (depth == 2 && (ch == '0' || ch == '1')) {
      current.push_back(ch - '0');
    }
  }
  if (rows.empty()) throw std::runtime_error("empty \"graph\" block in " + path);

  const unsigned n = (unsigned)rows.size();
  Graph g(n);
  for (unsigned i = 0; i < n; ++i) {
    if (rows[i].size() != n) {
      throw std::runtime_error("row " + std::to_string(i) + " of \"graph\" is not n long in " + path);
    }
    for (unsigned j = i + 1; j < n; ++j) {
      if (rows[i][j]) g.AddEdge(i, j);
    }
  }
  return g;
}

void Graph::SaveBaselineJson(const std::string& path) const {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write json file: " + path);
  out << "{\n\"size\": " << n_ << ",\n\"density\": " << Density() << ",\n";
  out << "\"graph\": [\n";
  for (unsigned i = 0; i < n_; ++i) {
    out << "[";
    for (unsigned j = 0; j < n_; ++j) {
      out << (IsJoined(i, j) ? 1 : 0);
      if (j + 1 != n_) out << ",";
    }
    out << (i + 1 != n_ ? "],\n" : "]\n");
  }
  out << "]\n}\n";
}

}  // namespace cc
