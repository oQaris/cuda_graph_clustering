// Раннер CPU-референса. CLI совпадает с CUDA-бинарником, поэтому оба сравнимы на одинаковых
// инстансах и параметрах.
#include "cc_cli.hpp"

int main(int argc, char** argv) {
  return cc::cli::Main(argc, argv, "cpu", &cc::SolveCpu);
}
