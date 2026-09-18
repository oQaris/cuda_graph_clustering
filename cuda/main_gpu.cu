// Раннер CUDA. CLI идентичен CPU-бинарнику.
#include "cc_cli.hpp"
#include "cc_gpu.hpp"

int main(int argc, char** argv) {
  return cc::cli::Main(argc, argv, "gpu", &cc::SolveGpu);
}
