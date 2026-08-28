# Two independent build paths:
#   make cpu      host reference + tests, plain C++17, no CUDA needed
#   make gpu      CUDA solver (needs nvcc)
#   make          both if nvcc is present, otherwise just cpu

CXX      ?= c++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Iinclude

NVCC     ?= nvcc
GPU_ARCH ?= sm_70
NVCCFLAGS ?= -O3 -std=c++17 -Iinclude --generate-code arch=compute_$(subst sm_,,$(GPU_ARCH)),code=$(GPU_ARCH)

BIN      := bin
HAVE_NVCC := $(shell command -v $(NVCC) 2>/dev/null)

CPU_SOURCES := src/cc_graph.cpp src/cc_pbils_cpu.cpp

# Optional bridge to the CPU baseline checkout, for objective cross-checking and
# head-to-head timing. Point BASELINE_DIR at your clone if it lives elsewhere.
BASELINE_DIR ?= ../../graph_correlation_clustering
BASELINE_SOURCES := \
  $(BASELINE_DIR)/src/clustering/BinaryClusteringVector.cpp \
  $(BASELINE_DIR)/src/clustering/TripleClusteringVector.cpp \
  $(BASELINE_DIR)/src/clustering/factories/BinaryClusteringFactory.cpp \
  $(BASELINE_DIR)/src/clustering/factories/TripleClusteringFactory.cpp \
  $(BASELINE_DIR)/src/graphs/AdjacencyMatrixGraph.cpp \
  $(BASELINE_DIR)/src/solvers/non_strict_two_correlation_clustering/common_functions/LocalSearch.cpp \
  $(BASELINE_DIR)/src/solvers/non_strict_two_correlation_clustering/ipls_algorithms/IPLSAlgorithm.cpp

.PHONY: all cpu gpu baseline emulate test clean
ifeq ($(HAVE_NVCC),)
all: cpu
	@echo "note: nvcc not found, GPU target skipped (run 'make gpu' on a CUDA host)"
else
all: cpu gpu
endif

cpu: $(BIN)/cc_cpu $(BIN)/cc_verify

$(BIN):
	@mkdir -p $(BIN)

$(BIN)/cc_cpu: $(CPU_SOURCES) src/main_cpu.cpp include/*.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $(CPU_SOURCES) src/main_cpu.cpp

$(BIN)/cc_verify: $(CPU_SOURCES) src/cc_verify.cpp include/*.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $(CPU_SOURCES) src/cc_verify.cpp

baseline: $(BIN)/cc_baseline

$(BIN)/cc_baseline: tools/baseline_ipls.cpp $(CPU_SOURCES) | $(BIN)
	@test -d $(BASELINE_DIR) || (echo "error: BASELINE_DIR=$(BASELINE_DIR) not found"; exit 1)
	$(CXX) -O2 -std=c++20 -Iinclude -I$(BASELINE_DIR)/include -o $@ tools/baseline_ipls.cpp $(CPU_SOURCES) $(BASELINE_SOURCES) -pthread

gpu: $(BIN)/cc_gpu

$(BIN)/cc_gpu: $(CPU_SOURCES) cuda/cc_pbils_gpu.cu cuda/main_gpu.cu include/*.hpp | $(BIN)
	$(NVCC) $(NVCCFLAGS) -o $@ cuda/cc_pbils_gpu.cu cuda/main_gpu.cu $(CPU_SOURCES)

test: $(BIN)/cc_verify
	./$(BIN)/cc_verify

clean:
	rm -rf $(BIN)
