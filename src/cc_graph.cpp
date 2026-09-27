#include "cc_graph.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <new>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "cc_math.hpp"
#include "cc_parallel.hpp"

namespace cc {
namespace {

constexpr uint64_t kErdosRenyiStream = 0xE4D05F;
constexpr uint64_t kSampleStream = 0x7A65;

std::string ReadWholeFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open file: " + path);
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

// Читает {"ключ": ["тег", ...], ...} и возвращает наборы тегов в порядке файла: номера тегов по возрастанию, без
// повторов (оригинал сравнивает std::set строк). Намеренно не общий парсер JSON: файлы датасетов устроены ровно так.
// Экранированный символ берётся как есть, этого хватает, чтобы различать теги.
std::vector<std::vector<int>> ReadTagSets(const std::string& path) {
  const std::string text = ReadWholeFile(path);
  size_t pos = 0;
  auto fail = [&](const char* what) {
    throw std::runtime_error(std::string("malformed tags json (") + what + ") at byte " + std::to_string(pos) + " of " +
                             path);
  };
  auto skip_space = [&]() {
    while (pos < text.size() && std::isspace((unsigned char)text[pos])) ++pos;
  };
  auto expect = [&](char ch) {
    skip_space();
    if (pos >= text.size() || text[pos] != ch) fail("unexpected character");
    ++pos;
  };
  auto next_is = [&](char ch) {
    skip_space();
    return pos < text.size() && text[pos] == ch;
  };
  auto read_string = [&](std::string& out) {
    expect('"');
    out.clear();
    while (pos < text.size() && text[pos] != '"') {
      if (text[pos] == '\\') ++pos;
      if (pos < text.size()) out += text[pos++];
    }
    if (pos >= text.size()) fail("unterminated string");
    ++pos;
  };

  std::unordered_map<std::string, int> tag_ids;
  std::vector<std::vector<int>> sets;
  std::string token;
  expect('{');
  if (next_is('}')) return sets;
  while (true) {
    read_string(token);  // ключ объекта графу не нужен: вершина — порядковый номер в файле
    expect(':');
    expect('[');
    std::vector<int> tags;
    if (!next_is(']')) {
      while (true) {
        read_string(token);
        const auto inserted = tag_ids.emplace(token, (int)tag_ids.size());
        tags.push_back(inserted.first->second);
        if (next_is(']')) break;
        expect(',');
      }
    }
    expect(']');
    std::sort(tags.begin(), tags.end());
    tags.erase(std::unique(tags.begin(), tags.end()), tags.end());
    sets.push_back(std::move(tags));
    if (next_is('}')) break;
    expect(',');
  }
  return sets;
}

// Мера сходства двух наборов тегов по размеру пересечения и размерам наборов. Формулы и порядок операций в double те
// же, что в TagsGraphFactory::Chance оригинала, поэтому и сравнение с порогом выходит тем же, включая пустые наборы
// (0/0 даёт NaN, и ребра нет).
enum class TagSimilarity { kJaccard, kCosine, kDice, kOverlap };

TagSimilarity ParseTagSimilarity(const std::string& name) {
  if (name == "jaccard") return TagSimilarity::kJaccard;
  if (name == "cosine") return TagSimilarity::kCosine;
  if (name == "dice") return TagSimilarity::kDice;
  if (name == "overlap") return TagSimilarity::kOverlap;
  throw std::runtime_error("unknown similarity '" + name + "', expected jaccard, cosine, dice or overlap");
}

double Similarity(TagSimilarity kind, size_t common, size_t size_a, size_t size_b) {
  const double inter = (double)common;
  const double a = (double)size_a;
  const double b = (double)size_b;
  switch (kind) {
    case TagSimilarity::kJaccard: return inter / (double)(size_a + size_b - common);
    case TagSimilarity::kCosine: return inter / std::sqrt(a * b);
    case TagSimilarity::kDice: return 2.0 * inter / (a + b);
    case TagSimilarity::kOverlap: return inter / std::min(a, b);
  }
  return 0.0;
}

// Граф без рёбер; нехватку памяти под битовую матрицу объясняет понятным сообщением.
Graph EmptyGraph(unsigned n, const char* hint) {
  try {
    return Graph(n);
  } catch (const std::bad_alloc&) {
    const double gib = (double)n * ((n + kWordBits - 1) / kWordBits) * sizeof(Word) / 1073741824.0;
    throw std::runtime_error("bit matrix of " + std::to_string(n) + " vertices needs " + std::to_string(gib) +
                             " GiB of host memory" + hint);
  }
}

size_t CommonCount(const std::vector<int>& a, const std::vector<int>& b) {
  size_t common = 0;
  for (size_t i = 0, j = 0; i < a.size() && j < b.size();) {
    if (a[i] < b[j]) {
      ++i;
    } else if (b[j] < a[i]) {
      ++j;
    } else {
      ++common;
      ++i;
      ++j;
    }
  }
  return common;
}

}  // namespace

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
  uint64_t state = StreamSeed(seed, kErdosRenyiStream, 0);
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

// Ищет ключ "graph" и читает следующую за ним матрицу 0/1 в скобках. Намеренно сканер, а не JSON-парсер: файлы
// результатов бейзлайна несут всю матрицу вперемешку с посторонними полями, а нужен только этот блок.
Graph Graph::LoadBaselineJson(const std::string& path) {
  const std::string text = ReadWholeFile(path);

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

// Граф строится не перебором пар, а по классам: у объектов с одинаковым набором тегов одинаковые строки матрицы (кроме
// бита на диагонали), поэтому строка считается один раз на класс и копируется каждому его объекту. Матрица выходит та
// же, что у попарного построения оригинала, но на всём датасете это секунды, а не часы.
Graph Graph::LoadTags(const std::string& path, const std::string& similarity, double threshold, unsigned n,
                      uint64_t seed) {
  const TagSimilarity kind = ParseTagSimilarity(similarity);
  const std::vector<std::vector<int>> sets = ReadTagSets(path);
  if (sets.empty()) throw std::runtime_error("no objects in " + path);

  const std::vector<unsigned> objects = SampleWithoutReplacement((unsigned)sets.size(), n, seed);
  const unsigned size = (unsigned)objects.size();

  // Классы — различные наборы тегов в порядке первой встречи; члены каждого класса лежат подряд.
  std::map<std::vector<int>, int> class_ids;
  std::vector<const std::vector<int>*> class_tags;
  std::vector<int> vertex_class(size);
  for (unsigned v = 0; v < size; ++v) {
    const auto inserted = class_ids.emplace(sets[objects[v]], (int)class_tags.size());
    if (inserted.second) class_tags.push_back(&inserted.first->first);
    vertex_class[v] = inserted.first->second;
  }
  const int classes = (int)class_tags.size();
  std::vector<unsigned> member_start(classes + 1, 0);
  for (unsigned v = 0; v < size; ++v) ++member_start[vertex_class[v] + 1];
  for (int c = 0; c < classes; ++c) member_start[c + 1] += member_start[c];
  std::vector<unsigned> members(size);
  std::vector<unsigned> cursor(member_start.begin(), member_start.end() - 1);
  for (unsigned v = 0; v < size; ++v) members[cursor[vertex_class[v]]++] = v;

  Graph g = EmptyGraph(size, "; take a sample with --n");
  const unsigned words = g.words_per_row_;
  WorkerPool pool(std::min(classes, HardwareThreads()));
  std::vector<unsigned long long> degree_sums(pool.Workers(), 0);
  std::vector<std::vector<char>> joined(pool.Workers(), std::vector<char>(classes));
  std::vector<std::vector<Word>> rows(pool.Workers(), std::vector<Word>(words));
  pool.Run(classes, [&](int c, int worker) {
    std::vector<char>& join = joined[worker];
    std::vector<Word>& row = rows[worker];
    const std::vector<int>& tags = *class_tags[c];
    for (int other = 0; other < classes; ++other) {
      const std::vector<int>& other_tags = *class_tags[other];
      join[other] = Similarity(kind, CommonCount(tags, other_tags), tags.size(), other_tags.size()) >= threshold;
    }
    std::fill(row.begin(), row.end(), 0u);
    unsigned long long ones = 0;
    for (unsigned u = 0; u < size; ++u) {
      if (join[vertex_class[u]]) {
        row[u / kWordBits] |= 1u << (u % kWordBits);
        ++ones;
      }
    }
    // Свой класс соединён с собой (сходство равного набора равно 1), но петель в графе нет.
    const unsigned long long degree = ones - (join[c] ? 1 : 0);
    for (unsigned i = member_start[c]; i < member_start[c + 1]; ++i) {
      const unsigned v = members[i];
      Word* dst = g.bits_.data() + (size_t)v * words;
      std::copy(row.begin(), row.end(), dst);
      dst[v / kWordBits] &= ~(1u << (v % kWordBits));
    }
    degree_sums[worker] += degree * (member_start[c + 1] - member_start[c]);
  });
  g.edges_ = std::accumulate(degree_sums.begin(), degree_sums.end(), 0ull) / 2;
  return g;
}

std::vector<unsigned> SampleWithoutReplacement(unsigned total, unsigned n, uint64_t seed) {
  std::vector<unsigned> items(total);
  std::iota(items.begin(), items.end(), 0u);
  if (n == 0 || n >= total) return items;
  uint64_t state = StreamSeed(seed, kSampleStream, 0);
  for (unsigned i = 0; i < n; ++i) std::swap(items[i], items[i + RandBelow(state, total - i)]);  // Фишер-Йетс
  items.resize(n);
  return items;
}

}  // namespace cc
