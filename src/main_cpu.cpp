// CPU reference runner. Same CLI as the CUDA binary, so the two can be compared
// on identical instances and identical parameters.
#include "cc_cli.hpp"

int main(int argc, char** argv) {
  return cc::cli::Main(argc, argv, "cpu", &cc::SolveCpu);
}
